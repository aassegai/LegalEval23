import torch
import numpy as np
import sklearn
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam
from tqdm.auto import tqdm



class Trainer:
    def __init__(self, config: dict):
        """
        Fits end evaluates given model with Adam optimizer.
         Hyperparameters are specified in `config`
        Possible keys are:
            - n_epochs: number of epochs to train
            - lr: optimizer learning rate
            - weight_decay: l2 regularization weight
            - device: on which device to perform training ("cpu" or "cuda")
            - verbose: whether to print anything during training
        :param config: configuration for `Trainer`
        """
        self.config = config
        self.n_epochs = config["n_epochs"]
        opt_state_dict = config.get('opt', None) # get the optimizer state dict, if present
        self.opt_cls = config.get('optimizer_cls', torch.optim.Adam)
        self.opt_params = config.get('optimizer_params', {})
        self.scheduler_cls = config.get('scheduler_cls', torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.scheduler_params = config.get('scheduler_params', {})
        self.model = None
        self.opt = None
        self.scheduler = None
        self.history = None
        self.loss_fn = CrossEntropyLoss()
        self.device = config["device"]
        self.verbose = config.get("verbose", True)

    def fit(self, model, train_loader, val_loader):
        """
        Fits model on training data, each epoch evaluates on validation data
        :param model: PyTorch model 
        :param train_loader: DataLoader for training data
        :param val_loader: DataLoader for validation data
        :return:
        """
        self.model = model.to(self.device)
        if self.opt is None:
            # self.opt = self.setup_opt_fn(self.model) #.to(self.device)
            self.opt = self.opt_cls(self.model.parameters(), **self.opt_params)
        if self.scheduler is None:
            # self.scheduler = self.setup_scheduler_fn(self.opt) #.to(self.device)
            self.scheduler = self.scheduler_cls(self.opt, **self.scheduler_params)
        for epoch in range(self.n_epochs):
            print(f'Epoch {epoch}')
            train_info = self._train_epoch(train_loader)
            val_info = self._val_epoch(val_loader)
        return self.model.eval()

    def _train_epoch(self, train_loader):
        self.model.train()
        if self.verbose:
            train_loader = tqdm(train_loader)
        
        running_train_loss = 0.
        running_train_f1 = 0.

        for batch in train_loader:
            self.model.zero_grad()
            input_ids, attention_masks, labels = batch['ids'], batch['mask'], batch['targets']
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            labels = labels.to(self.device)
            preds = self.model(input_ids, attention_masks)
            loss = self.loss_fn(preds, labels)
            # print(preds.shape)

            loss.backward()
            self.opt.step()

            # print(labels.shape)         
            preds_class = torch.sigmoid(preds).cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            preds_class = np.array([np.argmax(pred) for pred in preds_class])
            labels = np.array([np.argmax(label) for label in labels])
            # print(f'preds: {preds_class[0:20]}')
            # print(f'true: {labels[0:10]}')

            f1 = f1_score(labels, preds_class, average='weighted')          

            running_train_loss += loss.item()
            running_train_f1 += f1

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_f1 = running_train_f1 / len(train_loader)
        if self.verbose:
            print(f'Train loss = {epoch_train_loss:.3}, Train f1 = {epoch_train_f1:.3}')
            # train_loader.set_description(f"Epoch train loss={epoch_train_loss:.3}; Epoch train acc:{epoch_train_acc:.3}")

        return {"epoch_train_fq": epoch_train_f1, "epoch_train_loss": epoch_train_loss}


    def _val_epoch(self, val_loader):
        self.model.eval()
        # all_logits = []
        # all_labels = []
        if self.verbose:
            val_loader = tqdm(val_loader)

        running_val_loss = 0.
        running_val_f1 = 0.

        with torch.no_grad():
            for batch in val_loader:
                # print(batch)
                input_ids, attention_masks, labels = batch['ids'], batch['mask'], batch['targets']
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                labels = labels.to(self.device)

                preds = self.model(input_ids, attention_masks)
                loss = self.loss_fn(preds, labels)


                preds_class = torch.sigmoid(preds).cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                preds_class = np.array([np.argmax(pred) for pred in preds_class])
                labels = np.array([np.argmax(label) for label in labels])

                # print(preds_class.shape)
                # print(labels.shape)

                f1 = f1_score(labels, preds_class, average='weighted')             

                running_val_loss += loss.item()
                running_val_f1 += f1

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_f1 = running_val_f1 / len(val_loader)

        self.scheduler.step(metrics=epoch_val_loss)
        if self.verbose:
            print(f'Val loss = {epoch_val_loss:.3}, Val f1 = {epoch_val_f1:.3}')
            # val_loader.set_description(f"Epoch val loss={epoch_val_loss:.3}; Epoch val acc:{epoch_val_acc:.3}")
        return {"epoch_val_acc": epoch_val_f1, "epoch_val_loss": epoch_val_loss}

    def predict(self, test_loader):
        if self.model is None:
            raise RuntimeError("You should train the model first")
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids, attention_masks, labels = batch['ids'], batch['mask'], batch['targets']
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                preds = self.model.forward(input_ids.to(self.device), attention_masks.to(self.device))
                preds = torch.sigmoid(preds).cpu().detach().numpy()
                print(preds.shape)
                preds_class = np.array([np.argmax(pred) for pred in preds])
                predictions.extend(preds_class)
        return np.asarray(predictions)