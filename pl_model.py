import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from deepspeed.ops.adam import FusedAdam
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names

from utils import get_inputs_and_labels, load_model_and_tokenizer
import logging

# 创建 logger
logger = logging.getLogger(__name__)

# 设置日志等级（DEBUG, INFO, WARNING, ERROR, CRITICAL）
logger.setLevel(logging.DEBUG)

class LitContraCLM(pl.LightningModule):
    def __init__(self, trainer_args, loss_func_tok=None, loss_func_seq=None, 
                 loss_func_tok_word=None, num_nodes=1):
        super(LitContraCLM, self).__init__()
        self.save_hyperparameters(trainer_args)
        self.validation_outputs = []
        # Load Model and Tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(
            trainer_args.model_name, 
            pad_token_id=trainer_args.pad_token_id,
            dropout_layers=trainer_args.dropout_layers,
            dropout_p=trainer_args.dropout_p,
            functional_dropout=trainer_args.functional_dropout
        )
        trainable_params = [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]
        frozen_params = [(name, param) for name, param in self.model.named_parameters() if not param.requires_grad]

        logger.info(f"Total number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        logger.info(f"Trainable parameters: {sum(p.numel() for _, p in trainable_params)}")
        logger.info(f"Frozen parameters: {sum(p.numel() for _, p in frozen_params)}")
        self.trainer_args = trainer_args
        self.loss_func_tok = loss_func_tok
        self.loss_func_seq = loss_func_seq
        self.mle_loss = torch.nn.CrossEntropyLoss()
        self.vocab_size = self.model.config.vocab_size
        self.embed_dim = self.model.config.hidden_size
        self.num_nodes = num_nodes


    def setup(self, stage):
        if stage == 'fit':
            # Hyperparamters and Configuration
            self.dropout_p = self.trainer_args.dropout_p
            self.functional_dropout = self.trainer_args.functional_dropout
            self.pad_token_id = self.trainer_args.pad_token_id

            self.lr = self.trainer_args.lr
            self.weight_decay = self.trainer_args.weight_decay
            self.num_warmup_steps = self.trainer_args.warmup_steps
            self.num_epochs = self.trainer_args.max_epochs
            self.train_batch_size = self.trainer_args.train_batch_size
            self.num_train_examples = self.trainer_args.num_training_examples
            self.num_gpu_per_node = self.trainer_args.devices
            self.accumulate_grad_batches = self.trainer_args.accumulate_grad_batches

            if self.trainer_args.max_steps == -1:
                num_steps_per_epoch = self.num_train_examples // (self.num_gpu_per_node * self.num_nodes * self.accumulate_grad_batches)
                self.num_training_steps = self.num_epochs * num_steps_per_epoch
                print(f"steps_per_epoch: {num_steps_per_epoch}\t total_training_steps: {self.num_training_steps}.")
            else:
                self.num_training_steps = self.trainer_args.max_steps

            self.no_scheduling = self.trainer_args.no_scheduling
            self.world_size = self.trainer_args.devices * self.num_nodes
            # Loss Configuration
            self.loss = self.trainer_args.loss
            assert self.loss in ["MLE_Only", "ContraCLM", "ContraCLMTok", "ContraCLMSeq"], \
                f"Loss: `{self.loss}` is not supported!"


    def forward(self, input_ids, attention_mask=None):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        return logits, outputs.hidden_states


    def training_step(self, batch, batch_idx):
        # token_ids_list= batch['input_ids']
        # for l in token_ids_list:
        token_ids_list= batch['input_ids']  
        first_rows = [t[0] for t in token_ids_list]
        second_rows = [t[1] for t in token_ids_list]
        # import pdb;pdb.set_trace()

        # 使用 torch.stack 将它们堆叠为新的 Tensor
        first_tensor = torch.stack(first_rows, dim=0)
        second_tensor = torch.stack(second_rows, dim=0)
        # import pdb; pdb.set_trace()
        input_ids1, labels1, attention_mask1 = get_inputs_and_labels(
            first_tensor, pad_token_id=self.pad_token_id, mask_pad=True
        )
        input_ids2, labels2, attention_mask2 = get_inputs_and_labels(
            second_tensor, pad_token_id=self.pad_token_id, mask_pad=True
        )
        # uniq_tokens = torch.unique(input_ids)
        # all_tokens = torch.sum(attention_mask)
        # self.log("all_tokens_per_gpu", all_tokens, sync_dist=True)
        # self.log("unique_tokens_per_gpu", len(uniq_tokens), sync_dist=True)

        # first forward pass
        logits1, hidden_states1 = self(input_ids1, attention_mask=attention_mask1)
        logits2, hidden_states2 = self(input_ids2, attention_mask=attention_mask2)
        last_hidden_states1 = hidden_states1[-1]
        last_hidden_states2 = hidden_states2[-1]
        logits_combined = torch.cat((logits1, logits2), dim=1)
        labels_combined = torch.cat((labels1, labels2), dim=1)
        input_ids_combined = torch.cat((input_ids1, input_ids2), dim=1)
        attention_mask_combined = torch.cat((attention_mask1, attention_mask2), dim=1)

        # compute the MLE loss on all devices independently
        loss = self.mle_loss(logits_combined.view(-1, self.vocab_size), labels_combined.view(-1))
        self.log("Train/Loss/MLE", loss, sync_dist=True, on_step=True, prog_bar=True)

        # Original MLE
        if self.loss == "MLE_Only":
            return loss

        # get the dropout based augmentation either via the second forwarding pass or functional dropout
        if self.functional_dropout:
            last_hidden_states_orig = logits_combined
            last_hidden_states = F.dropout(last_hidden_states_orig, p=self.dropout_p)
            last_hidden_states_2 = F.dropout(last_hidden_states_orig, p=self.dropout_p)
        else:
            _, hidden_states_a = self(input_ids1, attention_mask=attention_mask1)
            _, hidden_states_b = self(input_ids2, attention_mask=attention_mask2)
            # _, hidden_states_2 = self(input_ids_combined, attention_mask=attention_mask_combined)
            last_hidden_states_1 = hidden_states_a[-1]
            last_hidden_states_2 = hidden_states_b[-1]

        # Token-level loss
        if self.loss == "ContraCLMTok" or self.loss == "ContraCLM":
            loss_tok_a = self.loss_func_tok(last_hidden_states1, last_hidden_states_1, attention_mask1)
            loss_tok_b = self.loss_func_tok(last_hidden_states2, last_hidden_states_2, attention_mask2)
            loss += loss_tok_a
            loss += loss_tok_b
            self.log(f"Train/Loss/TokCL", loss_tok_a+loss_tok_b , sync_dist=True, on_step=True, prog_bar=True)
        #    _, hidden_states_2 = self(input_ids_combined, attention_mask=attention_mask_combined)
        #    last_hidden_states_2 = hidden_states_2[-1]

        # Token-level loss
        #if self.loss == "ContraCLMTok" or self.loss == "ContraCLM":
        #    loss_tok = self.loss_func_tok(last_hidden_states, last_hidden_states_2, attention_mask_combined)
         #   loss += loss_tok
         #   self.log(f"Train/Loss/TokCL", loss_tok, sync_dist=True, on_step=True, prog_bar=True)

        # Sequence-level loss
        if self.loss == "ContraCLMSeq" or self.loss == "ContraCLM":
            # We use all_gather to gather representations from all GPUs. Since all_gather results are not part of
            # computational graph, we replace the current process's corresponding embeddings with original tensors
            if self.world_size > 1:
                all_attention_mask1 = self.all_gather(attention_mask1).flatten(start_dim=0, end_dim=1)
                all_attention_mask2 = self.all_gather(attention_mask2).flatten(start_dim=0, end_dim=1)
                all_hidden_feature_1 = self.all_gather(last_hidden_states1)
                all_hidden_feature_1[self.global_rank] = last_hidden_states1
                all_hidden_feature_1 = all_hidden_feature_1.flatten(start_dim=0, end_dim=1)

                all_hidden_feature_2 = self.all_gather(last_hidden_states2)
                all_hidden_feature_2[self.global_rank] = last_hidden_states2
                all_hidden_feature_2 = all_hidden_feature_2.flatten(start_dim=0, end_dim=1)
            else:
                all_attention_mask1 = input_ids1
                all_attention_mask2 = input_ids2
                all_hidden_feature_1 = last_hidden_states1
                all_hidden_feature_2 = last_hidden_states2
            loss_seq = self.loss_func_seq(all_hidden_feature_1, all_hidden_feature_2, 
                                          all_attention_mask1, all_attention_mask2)
            loss += loss_seq
            self.log(f"Train/Loss/SeqCL", loss_seq, rank_zero_only=True, on_step=True, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        eval_fct = torch.nn.CrossEntropyLoss()
        # token_ids = batch['input_ids']
        token_ids_list= batch['input_ids']  
        first_rows = [t[0] for t in token_ids_list]
        second_rows = [t[1] for t in token_ids_list]
        first_tensor = torch.stack(first_rows, dim=0)
        second_tensor = torch.stack(second_rows, dim=0)
        # input_ids, labels, attention_mask = get_inputs_and_labels(
        #     token_ids, pad_token_id=self.pad_token_id, mask_pad=True
        # )
        input_ids1, labels1, attention_mask1 = get_inputs_and_labels(
            first_tensor, pad_token_id=self.pad_token_id, mask_pad=True
        )
        input_ids2, labels2, attention_mask2 = get_inputs_and_labels(
            second_tensor, pad_token_id=self.pad_token_id, mask_pad=True
        )
        logits1, hidden_states1 = self(input_ids1, attention_mask=attention_mask1)
        logits2, hidden_states2 = self(input_ids2, attention_mask=attention_mask2)
        last_hidden_states1 = hidden_states1[-1]
        last_hidden_states2 = hidden_states2[-1]
        logits_combined = torch.cat((logits1, logits2), dim=1)
        labels_combined = torch.cat((labels1, labels2), dim=1)
        if self.world_size > 1:
            all_attention_mask1 = self.all_gather(attention_mask1).flatten(start_dim=0, end_dim=1)
            all_attention_mask2 = self.all_gather(attention_mask2).flatten(start_dim=0, end_dim=1)
            all_hidden_feature_1 = self.all_gather(last_hidden_states1)
            all_hidden_feature_1[self.global_rank] = last_hidden_states1
            all_hidden_feature_1 = all_hidden_feature_1.flatten(start_dim=0, end_dim=1)

            all_hidden_feature_2 = self.all_gather(last_hidden_states2)
            all_hidden_feature_2[self.global_rank] = last_hidden_states2
            all_hidden_feature_2 = all_hidden_feature_2.flatten(start_dim=0, end_dim=1)
        else:
            all_attention_mask1 = input_ids1
            all_attention_mask2 = input_ids2
            all_hidden_feature_1 = last_hidden_states1
            all_hidden_feature_2 = last_hidden_states2
        loss_seq = self.loss_func_seq(all_hidden_feature_1, all_hidden_feature_2, 
                                        all_attention_mask1, all_attention_mask2)
        # logits, _ = self(input_ids, attention_mask=attention_mask)
        loss = eval_fct(logits_combined.view(-1, self.vocab_size), labels_combined.view(-1))
        loss += loss_seq
        self.validation_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
    # Aggregate the saved losses
         val_loss = torch.stack(self.validation_outputs).mean()
         perplexity = torch.exp(val_loss)
         self.log("Valid/Loss/MLE", val_loss, sync_dist=True, on_epoch=True, prog_bar=True)
         self.log("Valid/Loss/Perplexity", perplexity, sync_dist=True, on_epoch=True, prog_bar=True)
    # Clear validation outputs to avoid memory issues
         self.validation_outputs.clear()
    #def validation_epoch_end(self, validation_step_outputs):
    #    val_loss = torch.stack(validation_step_outputs).mean()
    #    perplexity = torch.exp(val_loss)
    #    self.log("Valid/Loss/MLE", val_loss, sync_dist=True, on_epoch=True, prog_bar=True)
    #    SELF.LOG("vAlid/Loss/Perplexity", perplexity, sync_dist=True, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optim_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if n in decay_parameters
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]

   
