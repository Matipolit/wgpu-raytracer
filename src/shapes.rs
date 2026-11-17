//! Procedural triangle meshes for moving objects in training captures (about unit
//! size, centred on the origin): varied silhouettes, curvature, thin parts and holes.
use crate::raytracer::mesh::MeshVertex;
use nalgebra_glm as glm;

pub const NAMES: [&str; 6] = ["torus", "crate", "cylinder", "rock", "knot", "cone"];

/// Shape `index` (mod NAMES.len()) as vertices and triangles.
pub fn shape(index: usize) -> (Vec<MeshVertex>, Vec<[u32; 3]>) {
    let mut m = Builder::default();
    match index % NAMES.len() {
        0 => m.surface(48, 24, |u, v| {
            let (a, b) = (u * std::f32::consts::TAU, v * std::f32::consts::TAU);
            glm::vec3(
                (0.35 + 0.14 * b.cos()) * a.cos(),
                0.14 * b.sin(),
                (0.35 + 0.14 * b.cos()) * a.sin(),
            )
        }),
        1 => {
            // Six flat faces (separate vertices: sharp edges).
            for axis in 0..3 {
                for sign in [-1f32, 1.] {
                    m.surface(2, 2, |u, v| {
                        let (s, t) = (u - 0.5, v - 0.5);
                        let mut p = [0f32; 3];
                        p[axis] = 0.5 * sign;
                        p[(axis + 1) % 3] = s * sign;
                        p[(axis + 2) % 3] = t;
                        glm::vec3(p[0], p[1], p[2]) * 0.8
                    });
                }
            }
        }
        2 => {
            m.surface(40, 1, |u, v| {
                let a = u * std::f32::consts::TAU;
                glm::vec3(0.3 * a.cos(), v - 0.5, -0.3 * a.sin())
            });
            for y in [-0.5f32, 0.5] {
                m.surface(40, 1, |u, v| {
                    let a = u * std::f32::consts::TAU * y.signum();
                    glm::vec3(0.3 * v * a.cos(), y, -0.3 * v * a.sin())
                });
            }
        }
        3 => m.surface(40, 20, |u, v| {
            let (a, b) = (u * std::f32::consts::TAU, v * std::f32::consts::PI);
            let d = glm::vec3(b.sin() * a.cos(), b.cos(), b.sin() * a.sin());
            let bump = 1.
                + 0.18 * (3. * d.x + 1.).sin() * (4. * d.y).cos()
                + 0.1 * (7. * d.z + 2. * d.x).sin();
            d * 0.42 * bump
        }),
        4 => m.surface(160, 12, |u, v| {
            // (2, 3) torus knot tube.
            let curve = |s: f32| {
                let a = s * std::f32::consts::TAU;
                let r = 0.25 + 0.1 * (3. * a).cos();
                glm::vec3(r * (2. * a).cos(), 0.1 * (3. * a).sin(), r * (2. * a).sin())
            };
            let c = curve(u);
            let tangent = glm::normalize(&(curve(u + 1e-3) - curve(u - 1e-3)));
            let side = glm::normalize(&glm::cross(&tangent, &glm::vec3(0., 1., 0.)));
            let up = glm::cross(&side, &tangent);
            let b = v * std::f32::consts::TAU;
            c + (side * b.cos() + up * b.sin()) * 0.06
        }),
        _ => {
            m.surface(40, 1, |u, v| {
                let a = u * std::f32::consts::TAU;
                glm::vec3(
                    0.35 * (1. - v) * a.cos(),
                    v - 0.4,
                    -0.35 * (1. - v) * a.sin(),
                )
            });
            m.surface(40, 1, |u, v| {
                let a = -u * std::f32::consts::TAU;
                glm::vec3(0.35 * v * a.cos(), -0.4, -0.35 * v * a.sin())
            });
        }
    }
    (m.vertices, m.triangles)
}

#[derive(Default)]
struct Builder {
    vertices: Vec<MeshVertex>,
    triangles: Vec<[u32; 3]>,
}

impl Builder {
    /// A (u, v) grid patch. Normals from central differences of the surface (every
    /// patch here extends smoothly past its edges); degenerate points (poles, cap centres,
    /// apexes) use the direction from the origin. Materials are double-sided.
    fn surface(&mut self, nu: u32, nv: u32, f: impl Fn(f32, f32) -> glm::Vec3) {
        let base = self.vertices.len() as u32;
        for j in 0..=nv {
            for i in 0..=nu {
                let (u, v) = (i as f32 / nu as f32, j as f32 / nv as f32);
                let p = f(u, v);
                let e = 1e-3;
                let n = glm::cross(&(f(u + e, v) - f(u - e, v)), &(f(u, v + e) - f(u, v - e)));
                let n = if glm::length(&n) > 1e-10 {
                    glm::normalize(&n)
                } else if glm::length(&p) > 1e-6 {
                    glm::normalize(&p)
                } else {
                    glm::vec3(0., 1., 0.)
                };
                self.vertices.push(MeshVertex::new(
                    [p.x, p.y, p.z],
                    [n.x, n.y, n.z],
                    [u * 4., v * 4.],
                ));
            }
        }
        let row = nu + 1;
        for j in 0..nv {
            for i in 0..nu {
                let a = base + j * row + i;
                let (b, c, d) = (a + 1, a + row, a + row + 1);
                self.triangles.push([a, c, b]);
                self.triangles.push([b, c, d]);
            }
        }
    }
}
