import logging
import os
import gc
import time
import torch
import torch.nn as nn
import numpy as np
from hyades.config import Config
from hyades.apps.federated_learning.models import registry as models_registry
from hyades.apps.federated_learning.trainers import base
from hyades.apps.federated_learning.utils import optimizers


class Trainer(base.Trainer):
    def __init__(self, model=None):
        super().__init__()
        if model is None:
            model = models_registry.get()
        self.model = model

    def train_process(self, config, trainset, sampler, cut_layer=None):
        custom_train = getattr(self, "train_model", None)

        retry_time = 0
        while True:
            try:
                # if callable(custom_train):
                #     self.train_model(config, trainset, sampler.get(), cut_layer)
                # else:
                if True:
                    batch_size = config['batch_size']
                    epochs = config['epochs']

                    logging.info("[Client #%d] Loading the dataset.",
                                 self.client_id)
                    _train_loader = getattr(self, "train_loader", None)

                    if callable(_train_loader):
                        train_loader = self.train_loader(batch_size, trainset,
                                                         sampler.get(), cut_layer)
                    else:
                        train_loader = torch.utils.data.DataLoader(
                            dataset=trainset,
                            shuffle=False,
                            batch_size=batch_size,
                            sampler=sampler.get(),
                            drop_last=True,
                            num_workers=1  # to avoid unpickling error on dataloading
                        )

                    # each epoch logging for 4 batches
                    log_interval = max(1, len(train_loader) // 4)  # avoid hard-coding

                    self.model.to(self.device)
                    self.model.train()

                    _loss_criterion = getattr(self, "loss_criterion", None)
                    if callable(_loss_criterion):
                        loss_criterion = self.loss_criterion(self.model)
                    else:
                        loss_criterion = nn.CrossEntropyLoss()
                    get_optimizer = getattr(self, "get_optimizer",
                                            optimizers.get_optimizer)
                    optimizer = get_optimizer(self.model)
                    lr_schedule = None

                    # Following Oort's requirements
                    epoch_train_loss = 1e-4
                    loss_decay = 1e-2  # TODO: avoid magic numbers

                    for epoch in range(1, epochs + 1):
                        for batch_id, (examples,
                                       labels) in enumerate(train_loader):
                            if Config().app.data.datasource in ["FEMNIST", "MNIST"] \
                                    and "cnn" not in Config().app.trainer.model_name:
                                # one channel to three channels
                                examples = examples.repeat(1, 3, 1, 1)

                            examples, labels = examples.to(self.device), labels.to(
                                self.device)
                            optimizer.zero_grad()

                            model_name = Config().app.trainer.model_name
                            if 'albert' in model_name:
                                outputs = self.model(examples, labels=labels)
                                loss = outputs[0]
                            else:
                                if cut_layer is None:
                                    outputs = self.model(examples)
                                else:
                                    outputs = self.model.forward_from(
                                        examples, cut_layer)
                                loss = loss_criterion(outputs, labels)

                            loss.backward()
                            optimizer.step()

                            # Following Oort's requirements
                            if epoch == 1:
                                loss_list = loss.tolist()
                                if isinstance(loss_list, list):
                                    temp_loss = sum([l ** 2 for l in loss_list
                                                     ]) / float(len(loss_list))
                                else:
                                    temp_loss = loss.data.item()
                                    if 'albert' in model_name:
                                        temp_loss /= len(labels)

                                if epoch_train_loss == 1e-4:
                                    epoch_train_loss = temp_loss
                                else:
                                    epoch_train_loss = (1. - loss_decay) * epoch_train_loss \
                                                       + loss_decay * temp_loss

                            if lr_schedule is not None:
                                lr_schedule.step()

                            if batch_id % log_interval == 0:
                                    logging.info(
                                        "[Client #{} #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                        .format(self.client_id, os.getpid(), epoch, epochs,
                                                batch_id, len(train_loader), loss.data.item()))

                        if hasattr(optimizer, "params_state_update"):
                            optimizer.params_state_update()

            except Exception as training_exception:
                if isinstance(training_exception, RuntimeError):
                    # accommodate for "CUDA out of memory."
                    # TODO: avoid hard-coding
                    sleep_interval = 1
                    log_interval = 10 * sleep_interval
                    if retry_time % log_interval == 0:
                        logging.info(f"Training on client #{self.client_id} failed due to "
                                     f"RuntimeError: {str(training_exception)[:20]}. "
                                     f"Retrying the {retry_time + 1}-th time...")
                    time.sleep(sleep_interval)
                    retry_time += 1
                    self.switch_cuda_device(retry_time=retry_time)
                    continue
                else:  # other possible exception
                    logging.info("Training on client #%d failed.", self.client_id)
                    raise training_exception

            # break if succeeded
            self.set_a_shared_value(
                key=["model_utility"],
                value=np.sqrt(epoch_train_loss)
            )
            break

    def train(self, trainset, sampler, cut_layer=None):
        config = Config().app.trainer._asdict()
        self.train_process(config, trainset, sampler, cut_layer)

    def test_process(self, config, testset):
        self.model.to(self.device)
        self.model.eval()

        retry_time = 0
        while True:
            try:
                custom_test = getattr(self, "test_model", None)

                if callable(custom_test):
                    accuracy = self.test_model(config, testset)
                else:
                    test_loader = torch.utils.data.DataLoader(
                        testset, batch_size=config['batch_size'], shuffle=False, drop_last=True
                    )

                    correct = 0
                    total = 0
                    overall_loss = 0
                    with torch.no_grad():
                        for examples, labels in test_loader:
                            if Config().app.data.datasource in ["FEMNIST", "MNIST"] \
                                    and "cnn" not in Config().app.trainer.model_name:
                                # one channel to three channels
                                examples = examples.repeat(1, 3, 1, 1)

                            examples, labels = examples.to(self.device), labels.to(
                                self.device)

                            model_name = Config().app.trainer.model_name
                            if 'albert' in model_name:
                                outputs = self.model(examples, labels=labels)
                                loss_value = outputs[0].data.item()
                                overall_loss += loss_value
                            else:
                                outputs = self.model(examples)

                                _, predicted = torch.max(outputs.data, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()

                    if 'albert' in model_name:  # actually it is perplexity
                        overall_loss /= len(test_loader)
                        accuracy = np.exp(overall_loss)
                    else:
                        accuracy = correct / total

            except Exception as testing_exception:
                if isinstance(testing_exception, RuntimeError):
                    # accommodate for "CUDA out of memory."
                    # TODO: avoid hard-coding
                    sleep_interval = 1
                    log_interval = 10 * sleep_interval
                    if retry_time % log_interval == 0:
                        logging.info(f"Testing on client #{self.client_id} failed due to "
                                     f"RuntimeError: {str(testing_exception)[:20]}. "
                                     f"Retrying the {retry_time + 1}-th time...")
                    time.sleep(sleep_interval)
                    retry_time += 1
                    self.switch_cuda_device(retry_time=retry_time)
                    continue
                else:  # other possible exception
                    logging.info("Testing on client #%d failed.", self.client_id)
                    raise testing_exception

            # break if succeeded
            break

        self.model.cpu()
        return accuracy

    def test(self, testset) -> float:
        config = Config().app.trainer._asdict()
        accuracy = self.test_process(config, testset)
        return accuracy

    def server_test(self, testset):
        config = Config().app.trainer._asdict()

        retry_time = 0
        while True:
            try:
                model_to_test = self.model
                model_to_test.to(self.device)
                model_to_test.eval()

                custom_test = getattr(self, "test_model", None)

                if callable(custom_test):
                    return self.test_model(config, testset)

                test_loader = torch.utils.data.DataLoader(
                    testset, batch_size=config['batch_size'], shuffle=False,
                    drop_last=True
                )

                correct = 0
                total = 0
                overall_loss = 0.0

                model_name = Config().app.trainer.model_name
                with torch.no_grad():
                    for examples, labels in test_loader:
                        if Config().app.data.datasource in ["FEMNIST", "MNIST"] \
                                and "cnn" not in Config().app.trainer.model_name:
                            # one channel to three channels
                            examples = examples.repeat(1, 3, 1, 1)

                        examples, labels = examples.to(self.device), labels.to(
                            self.device)

                        if 'albert' in model_name:  # actually perplexity
                            outputs = model_to_test(examples, labels=labels)
                            loss_value = outputs[0].data.item()
                            overall_loss += loss_value
                        else:
                            outputs = model_to_test(examples)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                if 'albert' in model_name:  # actually it is perplexity
                    overall_loss /= len(test_loader)
                    accuracy = np.exp(overall_loss)
                else:
                    accuracy = correct / total

            except Exception as testing_exception:
                if isinstance(testing_exception, RuntimeError):
                    # accommodate for "CUDA out of memory."
                    # TODO: avoid hard-coding
                    sleep_interval = 1
                    log_interval = 10 * sleep_interval
                    if retry_time % log_interval == 0:
                        logging.info(f"Testing on the server failed due to "
                                     f"RuntimeError: {str(testing_exception)[:20]}. "
                                     f"Retrying the {retry_time + 1}-th time...")
                    time.sleep(sleep_interval)
                    retry_time += 1
                    self.switch_cuda_device(retry_time=retry_time)
                    continue
                else:  # other possible exception
                    logging.info("Testing on the server failed.", self.client_id)
                    raise testing_exception

            # break if succeeded
            break

        gc.collect()
        return accuracy
