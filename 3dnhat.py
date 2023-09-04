import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

def help():
    print('Control:\n1-2 X rotation 3-4 Y rotation 5-6 Z rotation\n7-8 X shift 9-0 Y shift up-down Z shift')

def find_coeffs(source_coords, target_coords, aux_points):
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0] * t[0], -s[0] * t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1] * t[0], -s[1] * t[1]])
    A = np.array(matrix, dtype=np.float32)
    B = np.array(source_coords).reshape(8)
    res = (np.linalg.inv(A.T @ A) @ A.T @ B).reshape(8)

    Tr = np.concatenate([res, [1]], dtype=np.float32).reshape(3, 3)
    aux_points = np.concatenate([aux_points, np.ones((len(aux_points), 1), dtype=np.float32)], axis=1)
    aux_points = (aux_points @ np.linalg.inv(Tr).T)
    return aux_points

def plot_estimated_n_hat(s, t, center):
    s2 = find_coeffs(s, t, s)
    n = np.cross(s2[1] - s2[0], s2[2] - s2[0])
    n = -n * n[2]
    n_hat = n / np.linalg.norm(n)
    vec = np.stack([center, center + n_hat], axis=0)
    ax.plot(vec[:, 0], vec[:, 1], vec[:, 2], color='m')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

def gen_rect(width, height, z):
    obj = np.array([[-width/2, -height/2, z], [-width/2, height/2, z],
                    [width/2, -height/2, z], [width/2, height/2, z]])
    return obj

def rotZ(alpha):
    r = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                  [np.sin(alpha), np.cos(alpha), 0],
                  [0, 0, 1.]])
    return r

def rotX(alpha):
    r = np.array([[1, 0, 0],
                  [0, np.cos(alpha), -np.sin(alpha)],
                  [0, np.sin(alpha), np.cos(alpha)]])
    return r

def rotY(alpha):
    r = np.array([[np.cos(alpha), 0, -np.sin(alpha)],
                  [0, 1, 0],
                  [np.sin(alpha), 0, np.cos(alpha)]])
    return r

def plot4plane(plane: np.array, color: str):
    assert plane.shape == (4, 3)
    X = plane[:, 0].reshape(2, 2)
    Y = plane[:, 1].reshape(2, 2)
    Z = plane[:, 2].reshape(2, 2)
    ax.plot_surface(X, Y, Z, color=color, linewidth=1, antialiased=False, alpha=0.5)

def plot_n_hat(plane):
    center = plane.mean(axis=0)
    n = np.cross(plane[0] - plane[1], plane[0] - plane[2])
    n = -n * n[2]
    n_hat = n / np.linalg.norm(n)
    vec = np.stack([center, center + n_hat], axis=0)
    ax.plot(vec[:, 0], vec[:, 1], vec[:, 2], color='b')

def draw_scene(alpha_X, alpha_Y, alpha_Z, shift_X, shift_Y, shift_Z):
    camera = np.array([0, 0, 0.])
    camera_plane = gen_rect(4, 4, 1)
    probe_rect = gen_rect(4.5, 1, 6)

    probe_rect = probe_rect @ rotX(alpha=alpha_X) @ rotY(alpha=alpha_Y) @ rotZ(alpha_Z) + np.array([[shift_X, shift_Y, shift_Z]])

    plot4plane(camera_plane, 'r')
    plot4plane(probe_rect, 'y')
    plot_n_hat(probe_rect)
    ax.scatter([0], [0], [0])

    proj = []
    for i in probe_rect:
        line = np.stack([camera, i], axis=0)
        ax.plot(line[:, 0], line[:, 1], line[:, 2], c='y')
        line_proj = line * (1/(line[1, 2]))
        ax.plot(line_proj[:, 0], line_proj[:, 1], line_proj[:, 2], c='r')
        proj.append(line_proj[1])

    proj = np.stack(proj)
    plot4plane(proj, 'y')

    plot_estimated_n_hat(gen_rect(4.5, 1, 6)[:, :2], proj[:, :2], center=(probe_rect * np.array([[0.2], [0.2], [0.3], [0.3]])).sum(axis=0))

# init state
state = {'alpha_X' : 0.4, 'alpha_Y' : 0.4, 'alpha_Z' : 0.0, 'shift_X' : 0.3, 'shift_Y' : 0.2, 'shift_Z' : -0.3}

# print help
help()

# initial scene draw
draw_scene(alpha_X=state['alpha_X'], alpha_Y=state['alpha_Y'], alpha_Z=state['alpha_Z'],
           shift_X=state['shift_X'], shift_Y=state['shift_Y'], shift_Z=state['shift_Z'])

# keypress
def keypress(event):
    print(event.key, 'pressed')
    if event.key == '1':
        state['alpha_X'] += 0.05
    elif event.key == '2':
        state['alpha_X'] -= 0.05
    elif event.key == '3':
        state['alpha_Y'] += 0.05
    elif event.key == '4':
        state['alpha_Y'] -= 0.05
    elif event.key == '5':
        state['alpha_Z'] += 0.05
    elif event.key == '6':
        state['alpha_Z'] -= 0.05
    elif event.key == '7':
        state['shift_X'] += 0.05
    elif event.key == '8':
        state['shift_X'] -= 0.05
    elif event.key == '9':
        state['shift_Y'] += 0.05
    elif event.key == '0':
        state['shift_Y'] -= 0.05
    elif event.key == 'up':
        state['shift_Z'] += 0.05
    elif event.key == 'down':
        state['shift_Z'] -= 0.05

    plt.cla()
    draw_scene(alpha_X=state['alpha_X'], alpha_Y=state['alpha_Y'], alpha_Z=state['alpha_Z'],
               shift_X=state['shift_X'], shift_Y=state['shift_Y'], shift_Z=state['shift_Z'])

    ax.set_aspect('equal')
    plt.draw()

fig.canvas.mpl_connect('key_press_event', keypress)

ax.set_aspect('equal')
plt.show()
plt.draw()