Modelo de Convecção Seca (Simple Cloud Model - Nickerson, 1965)
Baseado no Capítulo 6, Seção 5, pg. 139 do livro MCS

Domínio: 740m x 710m, dx=dz=10m → grade 74x71 pontos
Integração: dt=3s, até 600s (201 passos), saída a cada 120s
Esquema temporal: Matsuno (preditor-corretor)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os

# =============================================================
# PARÂMETROS DO MODELO
# =============================================================
nx   = 74          # pontos em x
nz   = 71          # pontos em z
dx   = 10.0        # espaçamento (m)
dz   = 10.0        # espaçamento (m)
dt   = 3.0         # passo de tempo (s)
nt   = 200         # número de passos (600s / 3s = 200 passos → t=0..600)
nout = 40          # salvar a cada 40 passos (120s)
nu   = 0.5         # difusividade (m²/s)
beta = 4.0e-3      # aquecimento diabático máximo (K/s)
theta0 = 300.0     # temperatura de referência (K)

# Coordenadas
x = np.arange(nx) * dx   # 0, 10, ..., 730 m
z = np.arange(nz) * dz   # 0, 10, ..., 700 m
X, Z = np.meshgrid(x, z, indexing='ij')

# =============================================================
# AQUECIMENTO DIABÁTICO Q(x,z)
# =============================================================
# Ativo entre x=[180,560]m e z=[80,120]m
# Forma cosseno: Q = beta * cos(π*(x-x0)/(2*Lx)) * cos(π*(z-z0)/(2*Lz))
def diabatic_heating(X, Z):
    Q = np.zeros((nx, nz))
    x0, Lx = 370.0, 190.0    # centro e metade do domínio em x (180–560)
    z0, Lz = 100.0,  20.0    # centro e metade do domínio em z (80–120)
    mask = (X >= 180) & (X <= 560) & (Z >= 80) & (Z <= 120)
    Q[mask] = beta * np.cos(np.pi * (X[mask] - x0) / (2 * Lx)) \
                   * np.cos(np.pi * (Z[mask] - z0) / (2 * Lz))
    return Q

Q_diab = diabatic_heating(X, Z)

# =============================================================
# CONDIÇÃO INICIAL: BOLHA QUENTE
# =============================================================
def initial_bubble(X, Z):
    """Bolha quente centrada em (xc, zc) com raio r"""
    xc, zc = 370.0, 260.0   # centro da bolha (m)
    r = 150.0                # raio (m)
    phi = np.zeros((nx, nz))
    dist2 = (X - xc)**2 + (Z - zc)**2
    inside = dist2 <= r**2
    phi[inside] = 2.0 * (1.0 - dist2[inside] / r**2)  # perfil parabólico (K)
    return phi

# =============================================================
# FUNÇÕES AUXILIARES: OPERADORES NUMÉRICOS
# =============================================================
def laplacian_2d(f):
    """Laplaciano centrado de ordem 2 (interior apenas)"""
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        f[2:,   1:-1] + f[:-2,  1:-1] +
        f[1:-1, 2:  ] + f[1:-1, :-2 ] -
        4.0 * f[1:-1, 1:-1]
    ) / (dx * dz)
    return lap

def jacobian(psi, zeta):
    """Jacobiano J(psi, zeta) - advecção de vorticidade"""
    J = np.zeros_like(psi)
    J[1:-1, 1:-1] = (
        (psi[2:, 1:-1] - psi[:-2, 1:-1]) * (zeta[1:-1, 2:] - zeta[1:-1, :-2]) -
        (psi[1:-1, 2:] - psi[1:-1, :-2]) * (zeta[2:, 1:-1] - zeta[:-2, 1:-1])
    ) / (4.0 * dx * dz)
    return J

def jacobian_phi(psi, phi):
    """Jacobiano J(psi, phi) - advecção de temperatura"""
    J = np.zeros_like(psi)
    J[1:-1, 1:-1] = (
        (psi[2:, 1:-1] - psi[:-2, 1:-1]) * (phi[1:-1, 2:] - phi[1:-1, :-2]) -
        (psi[1:-1, 2:] - psi[1:-1, :-2]) * (phi[2:, 1:-1] - phi[:-2, 1:-1])
    ) / (4.0 * dx * dz)
    return J

def buoyancy(phi):
    """Termo de flutuabilidade: g/theta0 * phi → força em vorticidade"""
    # d(phi)/dx centrado
    dphidx = np.zeros_like(phi)
    dphidx[1:-1, 1:-1] = (phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2.0 * dx)
    g = 9.81
    return (g / theta0) * dphidx

def psi_from_zeta_sor(zeta, psi_old, omega=1.8, tol=1e-4, maxiter=500):
    """
    Resolve ∇²ψ = ζ via SOR (Successive Over-Relaxation)
    Condições de contorno: ψ=0 nas bordas
    """
    psi = psi_old.copy()
    for _ in range(maxiter):
        psi_prev = psi.copy()
        for i in range(1, nx-1):
            for j in range(1, nz-1):
                psi[i, j] = (1 - omega) * psi[i, j] + omega / 4.0 * (
                    psi[i+1, j] + psi[i-1, j] +
                    psi[i, j+1] + psi[i, j-1] -
                    dx * dz * zeta[i, j]
                )
        err = np.max(np.abs(psi - psi_prev))
        if err < tol:
            break
    return psi

def psi_from_zeta_numpy(zeta):
    """
    Resolve ∇²ψ = ζ via FFT-DST (muito mais rápido que SOR)
    Condições de contorno homogêneas de Dirichlet
    """
    from scipy.fft import dst, idst
    # DST-I nos dois eixos
    rhs = zeta[1:-1, 1:-1].copy() * dx * dx  # assumindo dx=dz
    nxi = nx - 2
    nzi = nz - 2
    # Transformada em x
    tmp = dst(rhs, type=1, axis=0) / (2 * (nxi + 1))
    tmp = dst(tmp, type=1, axis=1) / (2 * (nzi + 1))
    # Autovalores
    ii = np.arange(1, nxi + 1)
    jj = np.arange(1, nzi + 1)
    II, JJ = np.meshgrid(ii, jj, indexing='ij')
    eigenval = 2 * np.cos(np.pi * II / (nxi + 1)) + 2 * np.cos(np.pi * JJ / (nzi + 1)) - 4.0
    # Divisão (evitar divisão por zero)
    eigenval = np.where(np.abs(eigenval) < 1e-12, 1e-12, eigenval)
    sol = tmp / eigenval
    # Transformada inversa
    sol = idst(sol, type=1, axis=1) * (nzi + 1)
    sol = idst(sol, type=1, axis=0) * (nxi + 1)
    psi = np.zeros((nx, nz))
    psi[1:-1, 1:-1] = sol
    return psi

def velocities(psi):
    """u = -∂ψ/∂z,  w = +∂ψ/∂x"""
    u = np.zeros((nx, nz))
    w = np.zeros((nx, nz))
    u[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dz)
    w[1:-1, 1:-1] =  (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dx)
    return u, w

# =============================================================
# INTEGRAÇÃO TEMPORAL - MATSUNO (preditor-corretor)
# =============================================================
def matsuno_step(zeta, phi, psi, t_current, t_total):
    """Um passo do esquema Matsuno"""
    # Aquecimento diabático: ativo apenas na primeira metade
    Q = Q_diab if t_current <= t_total / 2 else np.zeros((nx, nz))

    # --- PREDITOR ---
    J_zeta   = jacobian(psi, zeta)
    lap_z    = laplacian_2d(zeta)
    buoy     = buoyancy(phi)

    dzeta_dt = -J_zeta + nu * lap_z + buoy

    J_phi    = jacobian_phi(psi, phi)
    lap_phi  = laplacian_2d(phi)
    # d(phi)/dz para termo de estabilidade (aqui desprezado → modelo neutro)
    dphi_dt  = -J_phi + nu * lap_phi + Q

    zeta_p = zeta.copy()
    phi_p  = phi.copy()
    zeta_p[1:-1, 1:-1] += dt * dzeta_dt[1:-1, 1:-1]
    phi_p [1:-1, 1:-1] += dt * dphi_dt [1:-1, 1:-1]
    psi_p = psi_from_zeta_numpy(zeta_p)

    # --- CORRETOR ---
    J_zeta_p = jacobian(psi_p, zeta_p)
    lap_z_p  = laplacian_2d(zeta_p)
    buoy_p   = buoyancy(phi_p)
    dzeta_dt_p = -J_zeta_p + nu * lap_z_p + buoy_p

    J_phi_p   = jacobian_phi(psi_p, phi_p)
    lap_phi_p = laplacian_2d(phi_p)
    dphi_dt_p = -J_phi_p + nu * lap_phi_p + Q

    zeta_new = zeta.copy()
    phi_new  = phi.copy()
    zeta_new[1:-1, 1:-1] += dt * dzeta_dt_p[1:-1, 1:-1]
    phi_new [1:-1, 1:-1] += dt * dphi_dt_p [1:-1, 1:-1]
    psi_new = psi_from_zeta_numpy(zeta_new)

    return zeta_new, phi_new, psi_new

# =============================================================
# INICIALIZAÇÃO
# =============================================================
print("=" * 60)
print("  MODELO DE CONVECÇÃO SECA - Nickerson (1965)")
print("=" * 60)
print(f"  Grade: {nx} x {nz}  |  dx=dz={dx}m")
print(f"  dt={dt}s  |  Integração até {nt*dt:.0f}s")
print(f"  Esquema: Matsuno (preditor-corretor)")
print("=" * 60)

zeta = np.zeros((nx, nz))
phi  = initial_bubble(X, Z)
psi  = psi_from_zeta_numpy(zeta)   # ψ=0 inicialmente

# Armazenar saídas
output_times = []
output_psi   = []
output_zeta  = []
output_phi   = []
output_u     = []
output_w     = []

# Salvar t=0
u0, w0 = velocities(psi)
output_times.append(0.0)
output_psi.append(psi.copy())
output_zeta.append(zeta.copy())
output_phi.append(phi.copy())
output_u.append(u0)
output_w.append(w0)

# =============================================================
# LOOP TEMPORAL
# =============================================================
t = 0.0
for step in range(1, nt + 1):
    zeta, phi, psi = matsuno_step(zeta, phi, psi, t, nt * dt)
    t += dt

    if step % nout == 0:
        u, w = velocities(psi)
        output_times.append(t)
        output_psi.append(psi.copy())
        output_zeta.append(zeta.copy())
        output_phi.append(phi.copy())
        output_u.append(u.copy())
        output_w.append(w.copy())
        print(f"  t = {t:6.1f} s  |  max|ζ|={np.max(np.abs(zeta)):.4f}  "
              f"max|φ|={np.max(np.abs(phi)):.4f}  max|ψ|={np.max(np.abs(psi)):.4f}")

print("\n  Integração concluída!")

# =============================================================
# SALVAR BINÁRIOS (compatível com Fortran stream)
# =============================================================
os.makedirs("./bins", exist_ok=True)
for k, (t_val, ps, ze, ph, uu, ww) in enumerate(zip(
        output_times, output_psi, output_zeta, output_phi, output_u, output_w)):
    step_k = k * nout
    for name, arr in [("psi",ps),("zeta",ze),("phi",ph),("u",uu),("w",ww)]:
        fname = f"./bins/snapshot_{step_k:04d}_{name}.bin"
        arr.astype(np.float32).tofile(fname)

print(f"  Binários salvos em ./bins/  ({len(output_times)} tempos)")

# =============================================================
# VISUALIZAÇÃO
# =============================================================
# =============================================================
# VISUALIZAÇÃO
# =============================================================
ntimes = len(output_times)
fig = plt.figure(figsize=(20, 5 * ntimes), facecolor='white')

cmap_phi  = plt.cm.RdYlBu_r
cmap_psi  = plt.cm.PuOr
cmap_zeta = plt.cm.seismic

for k in range(ntimes):
    t_val = output_times[k]
    ps    = output_psi[k].T
    ze    = output_zeta[k].T
    ph    = output_phi[k].T
    uu    = output_u[k].T
    ww    = output_w[k].T

    # Skip every 4 points for quiver
    step_q = 4
    Xq = X[::step_q, ::step_q].T
    Zq = Z[::step_q, ::step_q].T
    Uq = uu[::step_q, ::step_q]
    Wq = ww[::step_q, ::step_q]

    ax1 = fig.add_subplot(ntimes, 3, k*3 + 1)
    vmax_ph = max(np.abs(ph).max(), 0.01)
    c1 = ax1.contourf(x, z, ph, levels=20, cmap=cmap_phi,
                      vmin=-vmax_ph, vmax=vmax_ph)
    ax1.contour(x, z, ps, levels=15, colors='black', linewidths=0.5, alpha=0.6)
    plt.colorbar(c1, ax=ax1, label='φ (K)', pad=0.01)

    ax1.set_title(
        f't = {t_val:.0f} s  —  Temperatura Normalizada φ',
        color='black',
        fontsize=10
    )

    ax1.set_xlabel('x (m)', color='black')
    ax1.set_ylabel('z (m)', color='black')
    ax1.tick_params(colors='black')
    ax1.set_facecolor('white')

    for sp in ax1.spines.values():
        sp.set_edgecolor('black')

    ax2 = fig.add_subplot(ntimes, 3, k*3 + 2)
    vmax_ze = max(np.abs(ze).max(), 0.001)

    c2 = ax2.contourf(x, z, ze, levels=20, cmap=cmap_zeta,
                      vmin=-vmax_ze, vmax=vmax_ze)

    ax2.contour(x, z, ps, levels=15,
                colors='black',
                linewidths=0.4,
                alpha=0.5)

    plt.colorbar(c2, ax=ax2, label='ζ (s⁻¹)', pad=0.01)

    ax2.set_title(
        f't = {t_val:.0f} s  —  Vorticidade ζ',
        color='black',
        fontsize=10
    )

    ax2.set_xlabel('x (m)', color='black')
    ax2.set_ylabel('z (m)', color='black')
    ax2.tick_params(colors='black')
    ax2.set_facecolor('white')

    for sp in ax2.spines.values():
        sp.set_edgecolor('black')

    ax3 = fig.add_subplot(ntimes, 3, k*3 + 3)

    speed = np.sqrt(uu**2 + ww**2)

    c3 = ax3.contourf(x, z, speed, levels=20, cmap='hot')

    ax3.quiver(Xq, Zq, Uq, Wq,
               color='black',
               alpha=0.7,
               scale=None,
               width=0.002)

    ax3.contour(x, z, ps, levels=10,
                colors='black',
                linewidths=0.4,
                alpha=0.4)

    plt.colorbar(c3, ax=ax3, label='|V| (m/s)', pad=0.01)

    ax3.set_title(
        f't = {t_val:.0f} s  —  Campo de Vento (u,w)',
        color='black',
        fontsize=10
    )

    ax3.set_xlabel('x (m)', color='black')
    ax3.set_ylabel('z (m)', color='black')
    ax3.tick_params(colors='black')
    ax3.set_facecolor('white')

    for sp in ax3.spines.values():
        sp.set_edgecolor('black')

fig.suptitle(
    'Modelo de Convecção Seca — Nickerson (1965)\n'
    'φ: temperatura normalizada  |  ζ: vorticidade  |  contornos: função de corrente ψ',
    color='black',
    fontsize=13,
    y=1.005
)

plt.tight_layout()

plt.savefig(
    './conveccao_seca_resultado.png',
    dpi=130,
    bbox_inches='tight',
    facecolor='white'
)

print("\n  Figura salva!")

# =============================================================
# FIGURA ESTILO ARTIGO #
# =============================================================
# Pegar o tempo final disponível (t=600s) ou o mais próximo
idx_ref = -1
t_ref = output_times[idx_ref]

fig2, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor='#f5f0e8')
fig2.patch.set_facecolor('#f5f0e8')

ph_ref = output_phi[idx_ref].T
ps_ref = output_psi[idx_ref].T
ze_ref = output_zeta[idx_ref].T

# Painel esquerdo: φ + ψ 
ax = axes[0]
ax.set_facecolor('white')
levels_phi = np.linspace(ph_ref.min(), ph_ref.max(), 15)
levels_psi = np.linspace(ps_ref.min(), ps_ref.max(), 20)
cs1 = ax.contour(x, z, ph_ref, levels=levels_phi, colors='black', linewidths=1.0)
cs2 = ax.contour(x, z, ps_ref, levels=levels_psi, colors='black',
                 linewidths=0.7, linestyles='dashed')
ax.clabel(cs1, inline=True, fontsize=7, fmt='%.2f')
ax.clabel(cs2, inline=True, fontsize=6, fmt='%.1f')
ax.set_xlabel('X (meters)', fontsize=11)
ax.set_ylabel('Z (meters)', fontsize=11)
ax.set_title(f'Fig. Bubble configuration at {t_ref:.0f} s\n'
             f'—: φ (excess pot. temp. °C)   ---: ψ (stream function m²/s)',
             fontsize=9)

# Painel direito: vorticidade colorida (estilo colorido do artigo)
ax2 = axes[1]
ax2.set_facecolor('white')
vmax = max(np.abs(ze_ref).max(), 0.001)
levels_ze = np.linspace(-vmax, vmax, 21)
cf = ax2.contourf(x, z, ze_ref, levels=levels_ze, cmap='coolwarm', alpha=0.85)
cs3 = ax2.contour(x, z, ze_ref, levels=levels_ze, colors='k', linewidths=0.4, alpha=0.5)
ax2.clabel(cs3, inline=True, fontsize=6, fmt='%.3f')
plt.colorbar(cf, ax=ax2, label='ζ (s⁻¹)', shrink=0.8)
ax2.set_xlabel('X (meters)', fontsize=11)
ax2.set_ylabel('Z (meters)', fontsize=11)
ax2.set_title(f'Vorticidade ζ at t = {t_ref:.0f} s', fontsize=9)

fig2.suptitle('Modelo de Convecção Seca — Nickerson (1965)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('./conveccao_seca_artigo.png',
            dpi=130, bbox_inches='tight', facecolor='#f5f0e8')
print("  Figura estilo artigo salva!")
plt.close('all')
print("\n  CONCLUÍDO.")
