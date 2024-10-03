# FRB cosmology reference

1. [[Deng, Wei 2014](https://ui.adsabs.harvard.edu/abs/2014ApJ...783L..35D/abstract)] FRB/GRB:
    $ H_0, \Omega_m, \Omega_\Lambda, DM_{IGM}(FRB), z(GRB)$ -> $f_{IGM}\Omega_b $ -> reionization history $ \chi_{e, He}, \chi_{e,H}$

2. [[Gao, He 2014](https://ui.adsabs.harvard.edu/abs/2014ApJ...788..189G/abstract)] FRB/GRB:
   $DM(FRB), z(GRB), H_0, \Omega_b, f_{IGM}$ -> Dark energy model $\omega$CDM

3. [[Wei, Jun-Jie](https://ui.adsabs.harvard.edu/abs/2018ApJ...860L...7W/abstract)] FRB/GW:
   $ z, (D_L(GW) \cdot DM_{IGM}(FRB))$ -> $ \Omega_m, \omega$ No need for $H_0$

# Detail in FRB $DM_{IGM} $

For FRB, cosmology distance is based on the dispersion measure (DM), which defined as
$$ DM=\int_0^{D_z}\frac{n_e(l)}{1+z(l)}dl $$

DM can be splitted into multiple terms:
$$ DM=DM_{MW}+DM{halo}+DM_{IGM}+\frac{DM_{host}+DM_{src}}{1+z}$$
where $DM_{MW}$ is the DM from the Milky Way, $DM_{halo}$ is the DM from the halo, $DM_{IGM}$ is the DM from the intergalactic medium, $DM_{host}$ is the DM from the host galaxy, and $DM_{src}$ is the DM from the source.

$DM_{MW}$ can be derived from model. $DM_{halo} \sim (30-80)pc \; cm^{-3}$

Most important is the IGM component:
$$ <DM_{IGM}>=\frac{3cH_0 \Omega_b }{8\pi G m_p}\int_0^z \frac{f_{IGM}\Chi(z')(1+z')dz'}{E(z')}$$
Here,
$$\chi=Y_H X_{e,H}+\frac{1}{2}Y_{He} X_{e,He}$$
Recall $Y_H=1/4$ and $Y_{He}=3/4$.

According to [[Bei Zhou 2014](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.89.107303)], $z\lesssim 3$, $X_{e,H}=X_{e,He}=1$; at $z\gtrsim 1.5$ $f_{IGM}\approx 90\%$; as $z\leq 0.4$ $f_{IGM}\approx (82\pm 4)\%$
