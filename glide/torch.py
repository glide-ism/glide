import torch
import cupy as cp

class GlideStep(torch.autograd.Function):

    @staticmethod
    def forward(ctx,t,dt,model,level,H_prev,bed,beta,smb):
        ctx.t = t
        ctx.dt = dt
        ctx.model = model
        ctx.level = level

        model.set_top_level(level)

        model.mg.state.H_prev.set(cp.asarray(H_prev.data),start_level=level)
        model.mg.state.H.set(cp.asarray(H_prev.data),start_level=level)
        model.mg.geometry.bed.set(cp.asarray(bed.data),start_level=level)
        model.mg.sliding.beta.set(cp.asarray(beta.data),start_level=level)
        model.mg.forcing.smb.set(cp.asarray(smb.data),start_level=level)

        model.forward(t,dt,update_geometry=False)

        u_torch = torch.tensor(model.mg[level].state.u.data)
        v_torch = torch.tensor(model.mg[level].state.v.data)
        H_torch = torch.tensor(model.mg[level].state.H.data)
        mask_torch = torch.tensor(model.mg[level].state.mask.data)
        phi_torch = torch.tensor(model.mg[level].state.phi.data)
        xi_torch = torch.tensor(model.mg[level].state.xi.data)

        ctx.save_for_backward(u_torch,v_torch,H_torch,mask_torch,phi_torch,xi_torch,H_prev,bed,beta,smb)
        ctx.mark_non_differentiable(mask_torch)
  
        return u_torch, v_torch, H_torch, mask_torch

    @staticmethod
    def backward(ctx, gu, gv, gH, gM):
        t = ctx.t
        dt = ctx.dt
        model = ctx.model
        level = ctx.level
        u_torch,v_torch,H_torch,mask_torch,phi_torch,xi_torch,H_prev,bed,beta,smb = ctx.saved_tensors

        model.mg.state.H_prev.set(cp.asarray(H_prev.data),start_level=level)
        model.mg.geometry.bed.set(cp.asarray(bed.data),start_level=level)
        model.mg.sliding.beta.set(cp.asarray(beta.data),start_level=level)
        model.mg.forcing.smb.set(cp.asarray(smb.data),start_level=level)

        model.mg.state.u.set(cp.asarray(u_torch.data),start_level=level)
        model.mg.state.v.set(cp.asarray(v_torch.data),start_level=level)
        model.mg.state.H.set(cp.asarray(H_torch.data),start_level=level)
        model.mg.state.phi.set(cp.asarray(phi_torch.data),start_level=level)
        model.mg.state.xi.set(cp.asarray(xi_torch.data),start_level=level)
        model.mg.state.mask.set(cp.asarray(mask_torch.data),start_level=level)

        converged = model.backward(t,dt,dJdu=cp.asarray(gu),dJdv=cp.asarray(gv),dJdH=cp.asarray(gH))


        g_H_prev = torch.tensor(model.mg[level].state.H_prev.grad)
        g_bed = torch.tensor(model.mg[level].geometry.bed.grad)
        g_beta = torch.tensor(model.mg[level].sliding.beta.grad)
        g_smb = torch.tensor(model.mg[level].forcing.smb.grad)
        
        #if not converged:
        #    g_H_prev[:,:] = 0.0
        #    g_bed[:,:] = 0.0
        #    g_beta[:,:] = 0.0
        #    g_smb[:,:] = 0.0

        return None, None, None, None, g_H_prev, g_bed, g_beta, g_smb

