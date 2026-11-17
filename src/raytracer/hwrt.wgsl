// Hardware ray queries (`--hw-rt`, tier B): one BLAS with two triangle geometries.
// Geometry 0 holds opaque double-sided triangles, flagged OPAQUE so traversal and
// intersection run entirely in fixed function hardware. Geometry 1 holds triangles
// that need the shader: alpha-tested (MASK) or single-sided. Their candidates run
// the same alpha test and back-face rule as the compute tracer.
// Primitive indices map back to triangle indices through two lists in meshData
// (header word 11: geometry 0, word 5: geometry 1).
@group(3) @binding(8) var meshAccel: acceleration_structure;

fn hwTriangle(geometry: u32, primitive: u32) -> u32 {
    return meshData[select(meshData[11u], meshData[5u], geometry == 1u) + primitive];
}

// Whether a geometry-1 candidate is a hit for the compute tracer's rules.
fn hwAccept(index: u32, u: f32, v: f32, frontFace: bool) -> bool {
    let accel = meshData[8u] + index * 12u;
    let flags = meshData[accel + 11u];
    if (flags & 1u) == 0u && !frontFace { return false; }
    if shadowRay && (flags & 8u) != 0u { return false; }
    if (flags & 2u) == 0u { return true; }
    let tri = meshData[2u] + index * 4u;
    let va = vertexOffset(meshData[tri]);
    let vb = vertexOffset(meshData[tri + 1u]);
    let vc = vertexOffset(meshData[tri + 2u]);
    let material = meshMaterialOffset(meshData[accel + 7u]);
    let weights = vec3(1. - u - v, u, v);
    let uv0 = weights.x * vertexUV0(va) + weights.y * vertexUV0(vb) + weights.z * vertexUV0(vc);
    let uv1 = weights.x * vertexUV1(va) + weights.y * vertexUV1(vb) + weights.z * vertexUV1(vc);
    let alpha = meshFloat(material + 3u) * meshTexture(material + 16u, uv0, uv1, false).a;
    return alpha >= meshFloat(material + 10u);
}

fn meshHwCandidate(ray: Ray, limit: f32, anyHit: bool, outBest: ptr<function, MeshCandidate>) -> bool {
    var rq: ray_query;
    let flags = select(RAY_FLAG_NONE, RAY_FLAG_TERMINATE_ON_FIRST_HIT, anyHit);
    rayQueryInitialize(&rq, meshAccel, RayDesc(flags, 0xffu, MIN_T, limit, ray.origin, ray.direction));
    while rayQueryProceed(&rq) {
        let c = rayQueryGetCandidateIntersection(&rq);
        let index = hwTriangle(c.geometry_index, c.primitive_index);
        if hwAccept(index, c.barycentrics.x, c.barycentrics.y, c.front_face != HW_FLIP_FACING) {
            rayQueryConfirmIntersection(&rq);
        }
    }
    let h = rayQueryGetCommittedIntersection(&rq);
    if h.kind == RAY_QUERY_INTERSECTION_NONE { return false; }
    *outBest = MeshCandidate(h.t, h.barycentrics.x, h.barycentrics.y, hwTriangle(h.geometry_index, h.primitive_index), 0u);
    return true;
}
