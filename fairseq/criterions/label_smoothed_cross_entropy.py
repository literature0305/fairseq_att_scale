# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, optimizer=None, sample_valid=None, stage_validation=False, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if stage_validation:
            model.eval()
            # validation stage
            ####################### without calibration #######################
            net_output = model(**sample["net_input"])
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            if self.report_accuracy:
                n_correct, total = self.compute_accuracy(model, net_output, sample)
                logging_output["n_correct"] = utils.item(n_correct.data)
                logging_output["total"] = utils.item(total.data)

            ####################### with temperature-scaling #######################
            type_calibration='temperature'
            net_output_valid_ts = model(**sample["net_input"], type_calibration=type_calibration)
            loss_temperature, nll_loss_temperature = self.compute_loss(model, net_output_valid_ts, sample, reduce=reduce)
            
            logging_output["loss_temperature"] = loss_temperature.data
            logging_output["nll_loss_temperature"] = nll_loss_temperature.data

            if self.report_accuracy:
                n_correct_valid_ts, total_valid_ts = self.compute_accuracy(model, net_output_valid_ts, sample)
                logging_output["n_correct_ts"] = utils.item(n_correct_valid_ts.data)
                logging_output["total_ts"] = utils.item(total_valid_ts.data)

            ####################### with att-weight-scaling #######################
            type_calibration='att_temp'
            net_output_valid_at = model(**sample["net_input"], type_calibration=type_calibration)
            loss_att_temp, nll_loss_att_temp = self.compute_loss(model, net_output_valid_at, sample, reduce=reduce)
            
            logging_output["loss_att_temp"] = loss_att_temp.data
            logging_output["nll_loss_att_temp"] = nll_loss_att_temp.data

            if self.report_accuracy:
                n_correct_valid_at, total_valid_at = self.compute_accuracy(model, net_output_valid_at, sample)
                logging_output["n_correct_at"] = utils.item(n_correct_valid_at.data)
                logging_output["total_at"] = utils.item(total_valid_at.data)

            ####################### with multi-head att-weight-scaling #######################
            type_calibration='mh_att_temp'
            net_output_valid_mh_at = model(**sample["net_input"], type_calibration=type_calibration)
            loss_mh_att_temp, nll_loss_mh_att_temp = self.compute_loss(model, net_output_valid_mh_at, sample, reduce=reduce)
            
            logging_output["loss_mh_att_temp"] = loss_mh_att_temp.data
            logging_output["nll_loss_mh_att_temp"] = nll_loss_mh_att_temp.data

            if self.report_accuracy:
                n_correct_valid_mh_at, total_valid_mh_at = self.compute_accuracy(model, net_output_valid_mh_at, sample)
                logging_output["n_correct_mh_at"] = utils.item(n_correct_valid_mh_at.data)
                logging_output["total_mh_at"] = utils.item(total_valid_mh_at.data)

            ####################### with bandwidth-scaling #######################
            type_calibration='band_width_scaling'
            net_output_valid_bw = model(**sample["net_input"], type_calibration=type_calibration)
            loss_bandwidth, nll_loss_bandwidth = self.compute_loss(model, net_output_valid_bw, sample, reduce=reduce)
            
            logging_output["loss_bandwidth"] = loss_bandwidth.data
            logging_output["nll_loss_bandwidth"] = nll_loss_bandwidth.data

            if self.report_accuracy:
                n_correct_valid_bw, total_valid_bw = self.compute_accuracy(model, net_output_valid_bw, sample)
                logging_output["n_correct_bw"] = utils.item(n_correct_valid_bw.data)
                logging_output["total_bw"] = utils.item(total_valid_bw.data)

            ####################### with multi-head bandwidth-scaling #######################
            type_calibration='mh_band_width_scaling'
            net_output_valid_mh_bw = model(**sample["net_input"], type_calibration=type_calibration)
            loss_mh_bw, nll_loss_mh_bw = self.compute_loss(model, net_output_valid_mh_bw, sample, reduce=reduce)
            
            logging_output["loss_mh_bw"] = loss_mh_bw.data
            logging_output["nll_loss_mh_bw"] = nll_loss_mh_bw.data

            if self.report_accuracy:
                n_correct_valid_mh_bw, total_valid_mh_bw = self.compute_accuracy(model, net_output_valid_mh_bw, sample)
                logging_output["n_correct_mh_bw"] = utils.item(n_correct_valid_mh_bw.data)
                logging_output["total_mh_bw"] = utils.item(total_valid_mh_bw.data)

            ####################### with ad-att-weight-scaling #######################
            type_calibration='ad_att_temp'
            net_output_valid_ad = model(**sample["net_input"], type_calibration=type_calibration)
            loss_ad_att_temp, nll_loss_ad_att_temp = self.compute_loss(model, net_output_valid_ad, sample, reduce=reduce)
            
            logging_output["loss_ad_att_temp"] = loss_ad_att_temp.data
            logging_output["nll_loss_ad_att_temp"] = nll_loss_ad_att_temp.data

            if self.report_accuracy:
                n_correct_valid_ad_at, total_valid_ad_at = self.compute_accuracy(model, net_output_valid_ad, sample)
                logging_output["n_correct_ad_at"] = utils.item(n_correct_valid_ad_at.data)
                logging_output["total_ad_at"] = utils.item(total_valid_ad_at.data)

            ####################### with mh-ad-att-weight-scaling #######################
            type_calibration='mh_ad_att_temp'
            net_output_valid_mh_ad = model(**sample["net_input"], type_calibration=type_calibration)
            loss_mh_ad_att_temp, nll_loss_mh_ad_att_temp = self.compute_loss(model, net_output_valid_mh_ad, sample, reduce=reduce)
            
            logging_output["loss_mh_ad_att_temp"] = loss_mh_ad_att_temp.data
            logging_output["nll_loss_mh_ad_att_temp"] = nll_loss_mh_ad_att_temp.data

            if self.report_accuracy:
                n_correct_valid_mh_ad_at, total_valid_mh_ad_at = self.compute_accuracy(model, net_output_valid_mh_ad, sample)
                logging_output["n_correct_mh_ad_at"] = utils.item(n_correct_valid_mh_ad_at.data)
                logging_output["total_mh_ad_at"] = utils.item(total_valid_mh_ad_at.data)

            if torch.randperm(50)[0] == 0:
                print('valid loss:', loss,'loss_temp:', loss_temperature,'loss_att_temp:', loss_att_temp,'loss_mh_att_temp:', loss_mh_att_temp, 'loss_bandwidth:', loss_bandwidth, 'loss_mh_bw:', loss_mh_bw, 'loss_ad_att_temp:', loss_ad_att_temp, 'loss_mh_ad_att_temp:', loss_mh_ad_att_temp)
                for name, param in model.named_parameters(): 
                    if 'scaling_factor' in name:
                        print('name:', name, 'param:', param)

            return loss, sample_size, logging_output
        else:
            model.train()
            ######################## stop gradient descent for scaling function ########################
            for name, param in model.named_parameters():
                if 'scaling_factor' in name:
                    param.requires_grad=False
                else:
                    param.requires_grad=True

            net_output = model(**sample["net_input"])
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            if self.report_accuracy:
                n_correct, total = self.compute_accuracy(model, net_output, sample)
                logging_output["n_correct"] = utils.item(n_correct.data)
                logging_output["total"] = utils.item(total.data)

            if optimizer is not None:
                with torch.autograd.profiler.record_function("backward"):
                    optimizer.backward(loss)

            if sample_valid is not None:
                # check validation set loss
                model.eval()
                type_calibration = 'None'
                net_output_tmp = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_tmp, nll_loss_tmp = self.compute_loss(model, net_output_tmp, sample_valid, reduce=reduce)

                ######################## temperature scaling ########################
                type_calibration = 'temperature'
                for name, param in model.named_parameters(): 
                    if 'scaling_factor_temp' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False

                net_output_ts = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_ts, nll_loss_valid_ts = self.compute_loss(model, net_output_ts, sample_valid, reduce=reduce)
                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_ts)

                ######################## attention weight scaling ########################
                type_calibration = 'att_temp'
                for name, param in model.named_parameters(): 
                    if 'scaling_factor_att_temp' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False

                net_output_at = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_at, nll_loss_valid_at = self.compute_loss(model, net_output_at, sample_valid, reduce=reduce)
                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_at)

                ######################## multi-head attention weight scaling ########################
                type_calibration = 'mh_att_temp'
                for name, param in model.named_parameters(): 
                    if 'scaling_factor_mh_att_temp' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False

                net_output_mh_at = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_mh_at, nll_loss_valid_mh_at = self.compute_loss(model, net_output_mh_at, sample_valid, reduce=reduce)
                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_mh_at)

                ######################## band width scaling ########################
                type_calibration = 'band_width_scaling'
                for name, param in model.named_parameters(): 
                    if 'scaling_factor_bw' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False

                net_output_bw = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_bw, nll_loss_valid_bw = self.compute_loss(model, net_output_bw, sample_valid, reduce=reduce)
                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_bw)

                ######################## multi-head band width scaling ########################
                type_calibration = 'mh_band_width_scaling'
                for name, param in model.named_parameters(): 
                    if 'scaling_factor_mh_bw' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False

                net_output_mh_bw = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_mh_bw, nll_loss_valid_mh_bw = self.compute_loss(model, net_output_mh_bw, sample_valid, reduce=reduce)
                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_mh_bw)

                ######################## adaptive attention weight scaling ########################
                type_calibration = 'ad_att_temp'
                for name, param in model.named_parameters(): 
                    if 'scaling_factor_ad_att_temp' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False

                net_output_ad_at = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_ad_at, nll_loss_valid_ad_at = self.compute_loss(model, net_output_ad_at, sample_valid, reduce=reduce)
                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_ad_at)

                ######################## adaptive attention weight scaling ########################
                type_calibration = 'mh_ad_att_temp'
                for name, param in model.named_parameters(): 
                    if 'scaling_factor_mh_ad_att_temp' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False

                net_output_mh_ad_at = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_mh_ad_at, nll_loss_valid_mh_ad_at = self.compute_loss(model, net_output_mh_ad_at, sample_valid, reduce=reduce)
                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_mh_ad_at)

                ### print ###
                if torch.randperm(500)[0] == 0:
                    print('validset train loss:', loss_tmp, 'train loss_ts:', loss_valid_ts, 'train loss_at', loss_valid_at,'loss_valid_mh_at:', loss_valid_mh_at, 'train loss_bw:', loss_valid_bw, 'loss_valid_mh_bw:', loss_valid_mh_bw, 'loss_valid_ad_at:', loss_valid_ad_at, 'loss_valid_ad_at:', loss_valid_ad_at, 'loss_valid_mh_ad_at:', loss_valid_mh_ad_at)
                    for name, param in model.named_parameters():
                        if 'scaling_factor' in name:
                            print('name:', name, 'param:', param)

        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
