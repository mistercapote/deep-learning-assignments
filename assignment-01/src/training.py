import torch
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu')


def bce_dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    inter = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)

    return F.binary_cross_entropy_with_logits(logits, targets) + (1 - dice.mean())


@torch.no_grad()
def compute_iou_dice(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (preds.sum(dim=1) + targets.sum(dim=1) + eps)
    return iou.sum(), dice.sum()


def train_model_binary(model, train_loader, val_loader, label, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    history = {'train_loss': [], 'val_iou': [], 'val_dice': []}
    best_iou = -1

    for epoch in range(epochs):
        model.train()
        accumulated_loss = torch.tensor(0.0, device=DEVICE)

        for images, masks, _, _ in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(images)
            loss = bce_dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            accumulated_loss += loss.detach() * images.size(0)
        train_loss = accumulated_loss.item() / len(train_loader.dataset)

        
        model.eval()
        total_iou = torch.tensor(0.0, device=DEVICE)
        total_dice = torch.tensor(0.0, device=DEVICE)

        with torch.no_grad():
            for images, masks, _, _ in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)
                logits = model(images)
                batch_iou, batch_dice = compute_iou_dice(logits, masks)
                total_iou += batch_iou
                total_dice += batch_dice

        val_iou = total_iou.item() / len(val_loader.dataset)
        val_dice = total_dice.item() / len(val_loader.dataset)
        scheduler.step(val_iou)

        history['train_loss'].append(train_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)
        print(f'Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}')

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), f'../models/best_model_{label}.pt')

    return history




# ==============================================================================
# 3. FUNÇÕES DE PERDA (LOSSES)
# ==============================================================================
def focal_loss(logits, targets, weight=None, gamma=2.0):
	logpt = F.log_softmax(logits, dim=1)
	ce = F.nll_loss(logpt, targets, weight=weight, reduction='none')
	pt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1).exp()
	focal_term = (1.0 - pt).clamp(min=0) ** gamma
	return (focal_term * ce).mean()


def combined_loss(logits, targets, pred_dist, targets_dist,
		weights,use_focal=True,dist_weight=1.0):
	if use_focal:
		l_cls = focal_loss(logits, targets, weight=weights)
	else:
		l_cls = F.cross_entropy(logits, targets, weight=weights)

	l_dist = F.l1_loss(pred_dist, targets_dist)
	total = l_cls + dist_weight * l_dist
	return total, l_cls.item(), l_dist.item()

# ==============================================================================
# 4. TREINAMENTO COM HISTÓRICO
# ==============================================================================
def train_model_ternary(model,train_loader,val_loader, 
						label=None, epochs=10,lr=1e-3,dist_weight=1.0):
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, mode='min', factor=0.5, patience=2
	)
	history = {
			'train_loss': [],
			'val_loss': [],
			'val_cls_loss': [],
			'val_dist_loss': [],
	}
	best_val_loss = float('inf')

	class_weights = torch.tensor([1.0, 1.0, 2.0])
	class_weights = class_weights.to(DEVICE)

	for epoch in range(epochs):
		model.train()
		running_train = torch.tensor(0.0, device=DEVICE)

		for images, labels, dists, _, _ in train_loader:
			images = images.to(DEVICE, non_blocking=True)
			labels = labels.to(DEVICE, non_blocking=True)
			dists = dists.to(DEVICE, non_blocking=True)

			optimizer.zero_grad()
			logits_cls, dist_pred = model(images)
			loss, _, _ = combined_loss(
			logits_cls, labels, dist_pred, dists, class_weights,
			use_focal=False,
			dist_weight=dist_weight
		)
			loss.backward()
			optimizer.step()
			running_train += loss.detach() * images.size(0)
		train_loss = running_train.item() / len(train_loader.dataset)

		model.eval()
		running_val = torch.tensor(0.0, device=DEVICE)
		running_cls = torch.tensor(0.0, device=DEVICE)
		running_dist = torch.tensor(0.0, device=DEVICE)
		
		with torch.no_grad():
			for images, labels, dists, _, _ in val_loader:
				images = images.to(DEVICE, non_blocking=True)
				labels = labels.to(DEVICE, non_blocking=True)
				dists = dists.to(DEVICE, non_blocking=True)

				logits_cls, dist_pred = model(images)
				loss, l_cls, l_dist = combined_loss(
						logits_cls,
						labels,
						dist_pred,
						dists,
						class_weights,
						dist_weight=dist_weight,
				)
				running_val += loss.detach() * images.size(0)
				running_cls += l_cls * images.size(0)
				running_dist += l_dist * images.size(0)

		n_val = len(val_loader.dataset)
		val_loss = running_val.item() / n_val
		val_cls = running_cls.item() / n_val
		val_dist = running_dist.item() / n_val

		scheduler.step(val_loss)
		history['train_loss'].append(train_loss)
		history['val_loss'].append(val_loss)
		history['val_cls_loss'].append(val_cls)
		history['val_dist_loss'].append(val_dist)

		print(
				f'Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val'
				f' Loss: {val_loss:.4f} (Cls: {val_cls:.4f}, Dist: {val_dist:.4f})'
		)

		if label != None:
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				torch.save(model.state_dict(), f'../models/best_model_{label}.pt')

	return history
