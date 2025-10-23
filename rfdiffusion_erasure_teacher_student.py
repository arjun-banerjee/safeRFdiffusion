"""
RFDiffusion Concept Erasure with Teacher-Student Architecture

This implementation correctly uses RFDiffusion's infrastructure with proper teacher-student setup
for concept erasure following the Genie methodology.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import logging
from pathlib import Path
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize_config_dir
from rfdiffusion.inference import utils as iu
from rfdiffusion.util import writepdb
import numpy as np
from rfdiffusion.contigs import ContigMap
import pickle
import json
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RFDiffusionErasureTeacherStudent:
    def __init__(
        self,
        config_path: str,
        config_name: str = "base",
        checkpoint_override: str = None,
        eta: float = 0.5,
        learning_rate: float = 1e-5,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0
    ):
        """Initialize fine-tuner with proper teacher-student architecture."""
        self.eta = eta
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        
        # Initialize Hydra config
        logger.info("Loading RFDiffusion configuration...")
        config_path = os.path.abspath(config_path)
        
        with initialize_config_dir(config_dir=config_path, version_base=None):
            self.conf = compose(config_name=config_name)
            
            # CRITICAL: Clear trb_save_ckpt_path before setting ckpt_override_path
            if hasattr(self.conf.inference, 'trb_save_ckpt_path'):
                self.conf.inference.trb_save_ckpt_path = None
            
            if checkpoint_override:
                self.conf.inference.ckpt_override_path = checkpoint_override
            
            self.conf.inference.num_designs = 1
            self.conf.inference.design_startnum = 0
            
        logger.info("Configuration loaded successfully")
        
        # Initialize teacher sampler (frozen)
        logger.info("Initializing RFDiffusion sampler for teacher...")
        self.teacher_sampler = iu.sampler_selector(self.conf)
        self.teacher_model = self.teacher_sampler.model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Initialize student sampler (trainable) - need fresh config
        logger.info("Initializing RFDiffusion sampler for student...")
        # Create a fresh configuration for student
        with initialize_config_dir(config_dir=config_path, version_base=None):
            student_conf = compose(config_name=config_name)
            
            # CRITICAL: Clear trb_save_ckpt_path before setting ckpt_override_path
            if hasattr(student_conf.inference, 'trb_save_ckpt_path'):
                student_conf.inference.trb_save_ckpt_path = None
            
            if checkpoint_override:
                student_conf.inference.ckpt_override_path = checkpoint_override
            
            student_conf.inference.num_designs = 1
            student_conf.inference.design_startnum = 0
            
        self.student_sampler = iu.sampler_selector(student_conf)
        self.student_model = self.student_sampler.model
        self.student_model.train()
        for param in self.student_sampler.model.parameters():
            param.requires_grad = True
        
        # Get device from model and set it for all components
        self.device = next(self.student_model.parameters()).device
        logger.info(f"Using device: {self.device}")
        
        # Setup optimizer for student only
        self.optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Setup scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.losses = []
        self.grad_norms = []
        self.eta_values = []
        
        logger.info(f"Fine-tuner initialized with teacher-student architecture")
        logger.info(f"Teacher: frozen, Student: trainable, eta={eta}")

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            return 1.0
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def prepare_batch(self, batch_data):
        """
        Prepare batch data using RFDiffusion's sampler infrastructure.
        
        Args:
            batch_data: Dict with 'pdb_path', 'contig_string', 'description'
        
        Returns:
            Prepared tensors for training
        """
        # Update sampler configuration with batch data
        pdb_path = batch_data['pdb_path'][0]  # Batch size 1
        contig_string = batch_data['contig_string'][0]
        
        return pdb_path, contig_string

    def compute_erasure_loss(self, pdb_path, contig_string, t):
        """
        Compute the Genie-style concept erasure loss with teacher–student setup.
        Fixed to maintain gradient flow through student model.
        """
        t_int = int(t.item()) if torch.is_tensor(t) else int(t)
        device = self.device
        
        # Parse PDB once
        parsed_pdb = iu.parse_pdb(pdb_path)

        def _fresh_sampler(base_conf, parsed_pdb, pdb_path, contig_string, trainable=False):
            from copy import deepcopy
            conf = deepcopy(base_conf)

            # Clear save path
            if hasattr(conf.inference, "trb_save_ckpt_path"):
                conf.inference.trb_save_ckpt_path = None

            # Set inputs
            conf.inference.input_pdb = pdb_path
            conf.contigmap.contigs = [contig_string]
            conf.inference.num_designs = 1

            sampler = iu.sampler_selector(conf)

            # Enforce contig AFTER initialization
            sampler._conf.contigmap.contigs = [contig_string]
            sampler.contig_conf.contigs = [contig_string]
            sampler._conf.inference.input_pdb = pdb_path
            sampler.contig_map = ContigMap(parsed_pdb=parsed_pdb, contigs=[contig_string])

            # Set trainability
            if trainable:
                sampler.model.train()
                for param in sampler.model.parameters():
                    param.requires_grad = True
            else:
                sampler.model.eval()
                for param in sampler.model.parameters():
                    param.requires_grad = False

            return sampler

        # ---------------------------------------------------------------------
        # 1. TEACHER CONDITIONAL (motif-conditioned) - NO GRADIENTS
        # ---------------------------------------------------------------------
        with torch.no_grad():
            teacher_cond = _fresh_sampler(self.conf, parsed_pdb, pdb_path, contig_string, trainable=False)
            
            x_init, seq_init = teacher_cond.sample_init()
            x_init, seq_init = x_init.to(device), seq_init.to(device)
            L = x_init.shape[0]

            atom_mask = torch.full((L, 14), False, dtype=torch.bool, device=device)
            atom_mask[:, :14] = True
            motif_mask = torch.argmax(seq_init, dim=-1) != 21

            diffusion_mask = ~motif_mask

            fa_stack, _ = teacher_cond.diffuser.diffuse_pose(
                x_init.cpu(), seq_init.cpu(), atom_mask.cpu(),
                diffusion_mask=diffusion_mask.cpu(), t_list=np.array([t_int])
            )
            x_t = (fa_stack[-1] if isinstance(fa_stack, list) else fa_stack.squeeze(0))[:, :14, :].to(device)

            px0_cond, _, _, _ = teacher_cond.sample_step(
                t=t_int,
                x_t=x_t.cpu(),
                seq_init=seq_init.cpu(),
                final_step=0,
                enable_grad=False
            )
            px0_cond = px0_cond.to(device).detach()  # Detach to prevent gradient flow

        # ---------------------------------------------------------------------
        # 2. TEACHER UNCONDITIONAL (all scaffold) - NO GRADIENTS
        # ---------------------------------------------------------------------
        with torch.no_grad():
            teacher_uncond = _fresh_sampler(self.conf, parsed_pdb, pdb_path, contig_string, trainable=False)
            
            x_init_u, seq_init_u = teacher_uncond.sample_init()
            x_init_u, seq_init_u = x_init_u.to(device), seq_init_u.to(device)
            diffusion_mask_u = torch.ones_like(motif_mask)

            fa_stack_u, _ = teacher_uncond.diffuser.diffuse_pose(
                x_init_u.cpu(), seq_init_u.cpu(), atom_mask.cpu(),
                diffusion_mask=diffusion_mask_u.cpu(), t_list=np.array([t_int])
            )
            x_t_u = (fa_stack_u[-1] if isinstance(fa_stack_u, list) else fa_stack_u.squeeze(0))[:, :14, :].to(device)

            px0_uncond, _, _, _ = teacher_uncond.sample_step(
                t=t_int,
                x_t=x_t_u.cpu(),
                seq_init=seq_init_u.cpu(),
                final_step=0,
                enable_grad=False
            )
            px0_uncond = px0_uncond.to(device).detach()  # Detach to prevent gradient flow

        # ---------------------------------------------------------------------
        # 3. STUDENT (trainable) - WITH GRADIENTS
        # ---------------------------------------------------------------------
        # Use the persistent student sampler but ensure it's properly configured
        self.student_sampler._conf.contigmap.contigs = [contig_string]
        self.student_sampler.contig_conf.contigs = [contig_string]
        self.student_sampler._conf.inference.input_pdb = pdb_path
        self.student_sampler.contig_map = ContigMap(parsed_pdb=parsed_pdb, contigs=[contig_string])
        
        # Re-initialize to get fresh x_init and seq_init for this configuration
        x_init_student, seq_init_student = self.student_sampler.sample_init()
        x_init_student = x_init_student.to(device)
        seq_init_student = seq_init_student.to(device)
        
        # Use the SAME diffused coordinates as teacher conditional for fair comparison
        # But ensure they require gradients for student forward pass
        x_t_student = x_t.clone().detach().requires_grad_(True)
        seq_init_student = seq_init.clone().detach()
        
        # Ensure student model is in training mode
        self.student_model.train()
        
        # Forward pass through student WITH gradient tracking
        px0_student, _, _, _ = self.student_sampler.sample_step(
            t=t_int,
            x_t=x_t_student.to(device),  # Already requires grad
            seq_init=seq_init_student.to(device),
            final_step=0,
            enable_grad=True  # Critical: enable gradients
        )
        
        # Ensure px0_student is on correct device and check gradient status
        px0_student = px0_student.to(device)
        
        # if not px0_student.requires_grad:
        #     raise RuntimeError("px0_student doesn't require gradients! Check model forward pass.")

        # ---------------------------------------------------------------------
        # 4. Loss Computation
        # ---------------------------------------------------------------------
        # Compute concept adjustment (detached teacher outputs)
        concept_adj = self.eta * (px0_cond - px0_uncond)
        target = px0_uncond - concept_adj

        # Option 1: Normalize to prevent explosion
        px0_uncond_norm = px0_uncond.norm(dim=-1, keepdim=True)
        target_norm = target.norm(dim=-1, keepdim=True)
        
        # Scale target to have similar magnitude as frozen_uncond
        scale_factor = px0_uncond_norm / (target_norm + 1e-8)
        scale_factor = torch.clamp(scale_factor, 0.1, 2.0)
        
        stable_target = target * scale_factor
        
        # Create scaffold mask (where we want to apply erasure), we apply loss over the conditioned mofits
        # Key: MOTIF_MASK covers the elements that are instructed to stay. We move the guess away from this direction
        # As such, 
        scaffold_mask = (motif_mask).float().unsqueeze(-1).unsqueeze(-1).to(device)

    
        # Compute masked loss
        element_loss = (px0_student - stable_target.detach()) ** 2 * 0.01  # Detach target
        masked_loss = element_loss * scaffold_mask
        valid = scaffold_mask.sum()
        
        if valid > 0:
            loss = masked_loss.sum() / valid
        else:
            RuntimeError
        # print("loss.requires_grad:", loss.requires_grad)
        # print("loss.grad_fn:", loss.grad_fn)
        # print("px0_student.requires_grad:", px0_student.requires_grad)
        # print("px0_student.grad_fn:", px0_student.grad_fn)


        
        # # Final check
        # if not loss.requires_grad:
        #     raise RuntimeError("Loss doesn't require gradients!")
        
        return loss


    
    def _ensure_device_consistency(self):
        """Helper method to ensure all model components are on the same device."""
        # Move student model to device
        self.student_model = self.student_model.to(self.device)
        
        # Move sampler components to device
        if hasattr(self.student_sampler, 'model'):
            self.student_sampler.model = self.student_sampler.model.to(self.device)
    
    def train_step(self, batch_data):
        """
        Perform single training step.
        
        Args:
            batch_data: Batch of training data
        
        Returns:
            Loss values
        """
        # Ensure device consistency before each step
        self._ensure_device_consistency()
        torch.autograd.set_detect_anomaly(True)

        
        self.optimizer.zero_grad()
        
        # Prepare batch
        pdb_path, contig_string = self.prepare_batch(batch_data)
        
        # Sample random timestep
        t = torch.randint(
            1,
            self.teacher_sampler.t_step_input,
            (1,),
            device=self.device
        )
        
        # Compute loss
        total_loss = self.compute_erasure_loss(pdb_path, contig_string, t)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.student_model.parameters(),
            self.max_grad_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'total_loss': total_loss.item(),
            'grad_norm': grad_norm.item(),
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def train(
        self,
        train_loader: DataLoader,
        num_steps: int,
        save_dir: str,
        log_interval: int = 10,
        save_interval: int = 100
    ):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            num_steps: Total training steps
            save_dir: Directory to save checkpoints
            log_interval: Steps between logging
            save_interval: Steps between checkpoint saves
        """
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Starting training for {num_steps} steps...")
        logger.info(f"Checkpoints will be saved to {save_dir}")
        
        step = 0
        epoch = 0
        
        while step < num_steps:
            epoch += 1
            logger.info(f"Epoch {epoch}")
            
            for batch_idx, batch_data in enumerate(train_loader):
                if step >= num_steps:
                    break
                
                # Training step
                losses = self.train_step(batch_data)
                self.losses.append(losses['total_loss'])
                self.grad_norms.append(losses['grad_norm'])
                self.eta_values.append(self.eta)
                
                # Logging
                if step % log_interval == 0:
                    logger.info(
                        f"Step {step}/{num_steps} | "
                        f"Total Loss: {losses['total_loss']:.4f} | "
                        f"Grad Norm: {losses['grad_norm']:.4f} | "
                        f"LR: {losses['lr']:.6f}"
                    )
                
                # Save checkpoint
                if step % save_interval == 0 and step > 0:
                    self.save_checkpoint(
                        os.path.join(save_dir, f"checkpoint_step_{step}.pt")
                    )
                
                step += 1
                self.global_step = step
                    
        
        # Save final checkpoint
        self.save_checkpoint(
            os.path.join(save_dir, "checkpoint_final.pt")
        )
        
        # Plot training curves
        self.plot_training_curves(save_dir)
        self.save_metrics(os.path.join(save_dir, "training_metrics.json"))
        
        logger.info("Training complete!")
    
    def save_checkpoint(self, path: str):
        """Save checkpoint with model and training state."""
        checkpoint = {
            'student_model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'losses': self.losses,
            'grad_norms': self.grad_norms,
            'eta_values': self.eta_values,
            'config': OmegaConf.to_container(self.conf, resolve=True),
            'eta': self.eta,
            'learning_rate': self.learning_rate,
            'device': str(self.device)
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint to resume training."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.student_model.load_state_dict(checkpoint['student_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.losses = checkpoint['losses']
        self.grad_norms = checkpoint['grad_norms']
        self.eta_values = checkpoint['eta_values']
        
        logger.info(f"Checkpoint loaded from {path}")
        logger.info(f"Resuming from step {self.global_step}")
    
    def plot_training_curves(self, save_dir):
        """Plot training metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curve
        ax1.plot(self.losses, alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        # Gradient norms
        ax2.plot(self.grad_norms, 'g-', alpha=0.7)
        ax2.axhline(y=self.max_grad_norm, color='r', linestyle='--', label='Clip threshold')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_title('Gradient Norms')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Loss distribution
        ax3.hist(self.losses, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Loss Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Loss Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Loss moving average
        if len(self.losses) > 20:
            window = 20
            ma = np.convolve(self.losses, np.ones(window)/window, mode='valid')
            ax4.plot(range(window-1, len(self.losses)), ma, 'r-', label=f'MA({window})')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Loss')
            ax4.set_title('Loss Moving Average')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
        plt.close()
        logger.info(f"Training curves saved to {save_dir}/training_curves.png")
    
    def save_metrics(self, path):
        """Save training metrics to JSON"""
        metrics = {
            'losses': self.losses,
            'grad_norms': self.grad_norms,
            'eta_values': self.eta_values,
        }
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {path}")


def create_simple_dataset(pdb_files, contig_strings, descriptions):
    """Create a simple dataset for training."""
    from torch.utils.data import Dataset
    
    class MotifDataset(Dataset):
        def __init__(self, pdb_files, contig_strings, descriptions):
            self.data = []
            for pdb, contig, desc in zip(pdb_files, contig_strings, descriptions):
                if os.path.exists(pdb):
                    self.data.append({
                        'pdb_path': pdb,
                        'contig_string': contig,
                        'description': desc
                    })
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    return MotifDataset(pdb_files, contig_strings, descriptions)


def collate_fn(batch):
    """Collate function for dataloader."""
    return {
        'pdb_path': [item['pdb_path'] for item in batch],
        'contig_string': [item['contig_string'] for item in batch],
        'description': [item['description'] for item in batch]
    }


def main():
    """Main training script."""
    
    # Configuration
    CONFIG_PATH = "config/inference"  # Path to RFDiffusion config directory
    CONFIG_NAME = "base"
    CHECKPOINT_PATH = "models/Base_ckpt.pt"
    OUTPUT_DIR = "finetuned_checkpoints"
    
    # Training hyperparameters
    ETA = 5
    LEARNING_RATE = 1e-6
    NUM_STEPS = 100
    BATCH_SIZE = 1
    
    # Check paths
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config path not found: {CONFIG_PATH}")
        logger.info("Make sure you're running from the RFDiffusion root directory")
        return
    
    if not os.path.exists(CHECKPOINT_PATH):
        logger.error(f"Checkpoint not found: {CHECKPOINT_PATH}")
        logger.info("Download with: bash scripts/download_models.sh")
        return
    
    # Create dataset
    logger.info("Creating training dataset...")
    
    pdb_files = [
        "examples/input_pdbs/5TPN.pdb",
    ]
    
    contig_strings = [
        "1/A80-90",
    ]
    
    descriptions = [
        "Alpha helix motif scaffolding",
    ]
    
    # Filter existing files
    existing_data = [
        (pdb, contig, desc)
        for pdb, contig, desc in zip(pdb_files, contig_strings, descriptions)
        if os.path.exists(pdb)
    ]
    
    if not existing_data:
        logger.error("No PDB files found!")
        return
    
    pdb_files, contig_strings, descriptions = zip(*existing_data)
    
    dataset = create_simple_dataset(
        list(pdb_files),
        list(contig_strings),
        list(descriptions)
    )
    
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize fine-tuner
    logger.info("Initializing teacher-student fine-tuner...")
    try:
        fine_tuner = RFDiffusionErasureTeacherStudent(
            config_path=CONFIG_PATH,
            config_name=CONFIG_NAME,
            checkpoint_override=CHECKPOINT_PATH,
            eta=ETA,
            learning_rate=LEARNING_RATE,
            warmup_steps=20,
            max_grad_norm=10
        )
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Train
    logger.info("Starting training...")
    try:
        fine_tuner.train(
            train_loader=train_loader,
            num_steps=NUM_STEPS,
            save_dir=OUTPUT_DIR,
            log_interval=10,
            save_interval=50
        )
        
        logger.info(f"\nTraining complete!")
        logger.info(f"Final checkpoint: {OUTPUT_DIR}/checkpoint_final.pt")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
