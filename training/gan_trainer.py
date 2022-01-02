import os
import numpy as np
import jittor as jt
from .gan_model import GANModel


class GANTrainer:
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """
    def __init__(self, opt):
        self.opt = opt

        self.gan_model = GANModel(opt)

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.gan_model.create_optimizers(opt)
            self.gan_model.create_loss_fns(opt)
            self.gan_model.set_requires_grad(False, False)

            if opt.resume_iter is not None:
                self.load(opt.resume_iter)

            self.g_losses = {}
            self.d_losses = {}
            self.trackables = {}
            self.interm_imgs = {}
            self.reports = {}
            self.set_fixed_noise()

    def run_generator_one_step(self, data):
        g_losses, generated = self.gan_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        self.optimizer_G.step(g_loss)
        self.generated = generated
        update_dict(self.g_losses, g_losses)

    def run_generator_regularization_one_step(self, data):
        output = self.gan_model(data, mode='generator-regularize')
        g_reg_losses, trackables = output
        g_reg_loss = sum(g_reg_losses.values()).mean()
        self.optimizer_G.step(g_reg_loss)
        update_dict(self.g_losses, g_reg_losses)
        update_dict(self.trackables, trackables)

    def run_discriminator_one_step(self, data):
        d_losses, interm_imgs = self.gan_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        self.optimizer_D.step(d_loss)
        update_dict(self.d_losses, d_losses)
        update_dict(self.interm_imgs, interm_imgs)

    def run_discriminator_regularization_one_step(self, data):
        d_reg_losses = self.gan_model(data, mode='discriminator-regularize')
        d_reg_loss = sum(d_reg_losses.values()).mean()
        self.optimizer_D.step(d_reg_loss)
        update_dict(self.d_losses, d_reg_losses)

    def train_one_step(self, data, iters):
        self.gan_model.set_requires_grad(False, True)
        self.run_discriminator_one_step(data)
        if not self.opt.no_d_regularize and iters % self.opt.d_reg_every == 0:
            self.run_discriminator_regularization_one_step(data)
        self.gan_model.set_requires_grad(True, False)
        self.run_generator_one_step(data)

    def get_latest_losses(self):
        self.reports = {**self.g_losses, **self.d_losses, **self.trackables}
        return {k: v.mean().item() for k, v in self.reports.items()}

    def get_latest_generated(self):
        return self.generated

    def get_visuals(self):
        visuals = {}
        sample, transf = self.gan_model.inference(self.sample_z, with_tf=True)
        interp = self.gan_model.inference(self.interp_z)

        sample_trunc = self.gan_model.inference(self.sample_z, trunc_psi=0.5)
        interp_trunc = self.gan_model.inference(self.interp_z, trunc_psi=0.5)

        def make_grid(sample: jt.Var, nrow):
            return jt.misc.make_grid(sample, nrow=nrow)

        visuals['sample'] = make_grid(sample, nrow=8)
        visuals['sample_transf'] = make_grid(transf, nrow=8)
        visuals['interp'] = make_grid(interp, nrow=8)
        visuals['sample_psi0.5'] = make_grid(sample_trunc, nrow=8)
        visuals['interp_psi0.5'] = make_grid(interp_trunc, nrow=8)
        update_dict(visuals, self.interm_imgs)
        return visuals

    def get_gan_model(self):
        return self.gan_model

    def save(self, iters):
        self.gan_model.save(iters)

        misc = {
            "g_optim": self.optimizer_G.state_dict(),
            "d_optim": self.optimizer_D.state_dict(),
            "opt": self.opt,
        }
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, f"{iters}_net_")
        jt.save(misc, save_path + "misc.jt")

    def load(self, iters):
        print(f"Resuming model at iteration {iters}")
        self.gan_model.load(iters)
        load_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, f"{iters}_net_")
        state_dict = jt.load(load_path + "misc.jt")
        self.optimizer_G.load_state_dict(state_dict["g_optim"])
        self.optimizer_D.load_state_dict(state_dict["d_optim"])

    def set_fixed_noise(self):
        os.makedirs('./cache_files/', exist_ok=True)
        if self.opt.reduce_visuals:
            sample_z_file = './cache_files/sample_z_reduced.jt'
            interp_z_file = './cache_files/interp_z_reduced.jt'
        else:
            sample_z_file = './cache_files/sample_z.jt'
            interp_z_file = './cache_files/interp_z.jt'

        if os.path.exists(sample_z_file):
            self.sample_z = jt.load(sample_z_file)
        else:
            if self.opt.reduce_visuals:
                z = jt.randn(8, 512)
            else:
                z = jt.randn(32, 512)

            jt.save(z, sample_z_file)
            self.sample_z = z

        if os.path.exists(interp_z_file):
            self.interp_z = jt.load(interp_z_file)
        else:
            with jt.no_grad():
                if self.opt.reduce_visuals:
                    z0 = jt.randn(1, 1, 512)
                    z1 = jt.randn(1, 1, 512)
                else:
                    z0 = jt.randn(4, 1, 512)
                    z1 = jt.randn(4, 1, 512)

                z = []
                for c in np.linspace(0, 1, 8):
                    z.append((1 - c) * z0 + c * z1)
                z = jt.concat(z, 1).view(-1, 512)
            jt.save(z, interp_z_file)
            self.interp_z = z


def update_dict(old_dict, new_dict):
    for key in new_dict.keys():
        if type(new_dict[key]) == jt.Var:
            new_dict[key] = new_dict[key].detach()
        old_dict[key] = new_dict[key]
