// Mesh buffer header: node/triangle counts, triangle/vertex/material/texture
// word offsets, logical sphere/light counts. BVH nodes begin at word 16.
@group(3) @binding(4) var<storage, read> meshData: array<u32>;
@group(3) @binding(5) var<storage, read> meshPixels: array<u32>;

fn meshFloat(offset: u32) -> f32 { return bitcast<f32>(meshData[offset]); }
fn meshVec2(offset: u32) -> vec2<f32> { return vec2(meshFloat(offset), meshFloat(offset+1u)); }
fn meshVec3(offset: u32) -> vec3<f32> { return vec3(meshFloat(offset),meshFloat(offset+1u),meshFloat(offset+2u)); }
fn meshVec4(offset: u32) -> vec4<f32> { return vec4(meshVec3(offset),meshFloat(offset+3u)); }
fn meshMaterialOffset(index: u32) -> u32 { return meshData[4u]+48u*index; }

// Integer address wrapping also wraps each bilinear footprint across seams.
fn meshWrap(x: i32, size: i32, mode: u32) -> u32 {
    if mode == 1u { return u32(clamp(x,0,size-1)); }
    if mode == 2u {
        let period = 2*size;
        let p = ((x % period)+period)%period;
        return u32(select(period-1-p,p,p<size));
    }
    return u32(((x % size)+size)%size);
}
fn meshTexel(desc: u32, point: vec2<i32>, srgb: bool) -> vec4<f32> {
    let x=meshWrap(point.x,i32(meshData[desc+1u]),meshData[desc+3u]&3u);
    let y=meshWrap(point.y,i32(meshData[desc+2u]),(meshData[desc+3u]>>2u)&3u);
    var value=unpack4x8unorm(meshPixels[meshData[desc]+y*meshData[desc+1u]+x]);
    if srgb {
        let low=value.rgb/12.92;
        let high=pow((value.rgb+vec3(0.055))/1.055,vec3(2.4));
        value=vec4(select(high,low,value.rgb<=vec3(0.04045)),value.a);
    }
    return value;
}
fn meshUV(slot: u32, uv0: vec2<f32>, uv1: vec2<f32>) -> vec2<f32> {
    let uv=select(uv0,uv1,meshData[slot+1u]==1u)*meshVec2(slot+4u);
    let angle=meshFloat(slot+6u);
    return meshVec2(slot+2u)+vec2(cos(angle)*uv.x-sin(angle)*uv.y,sin(angle)*uv.x+cos(angle)*uv.y);
}
fn meshTexture(slot: u32, uv0: vec2<f32>, uv1: vec2<f32>, srgb: bool) -> vec4<f32> {
    let index=meshData[slot];
    if index==0xffffffffu { return vec4(1.); }
    var desc=meshData[5u]+index*4u;
    if RAY_CONES && (srgb || (slot-meshData[4u])%48u!=16u) {
        let header=meshData[12u]+(index*2u+select(0u,1u,srgb))*4u;
        let levels=meshData[header+1u];
        let gradient=select(footprintUV0,footprintUV1,meshData[slot+1u]==1u)*abs(meshVec2(slot+4u));
        let angle=meshFloat(slot+6u);let c=abs(cos(angle));let s=abs(sin(angle));
        let rotated=vec2(c*gradient.x+s*gradient.y,s*gradient.x+c*gradient.y);
        let size=vec2(f32(meshData[desc+1u]),f32(meshData[desc+2u]));
        let lod=u32(clamp(floor(log2(max(max(rotated.x*size.x,rotated.y*size.y),1.))),0.,f32(levels-1u)));
        desc=meshData[15u]+(meshData[header]+lod)*4u;
    }
    let dimensions=vec2(f32(meshData[desc+1u]),f32(meshData[desc+2u]));
    let uv=meshUV(slot,uv0,uv1);
    // glTF image and UV origins are both top-left. No legacy sphere V flip.
    if (meshData[desc+3u]&16u)!=0u { return meshTexel(desc,vec2<i32>(floor(uv*dimensions)),srgb); }
    let coordinate=uv*dimensions-vec2(0.5);
    let point=vec2<i32>(floor(coordinate));
    let f=fract(coordinate);
    if STOCHASTIC_TEXTURE==2u || (STOCHASTIC_TEXTURE==1u && srgb) {
        let offset=vec2<i32>(select(0,1,textureRandom()<f.x),select(0,1,textureRandom()<f.y));
        return meshTexel(desc,point+offset,srgb);
    }
    return mix(mix(meshTexel(desc,point,srgb),meshTexel(desc,point+vec2(1,0),srgb),f.x),
               mix(meshTexel(desc,point+vec2(0,1),srgb),meshTexel(desc,point+vec2(1,1),srgb),f.x),f.y);
}

// Reciprocal directions are invariant throughout traversal. Parallel slabs use
// finite arithmetic and explicit rejection, including origins on box boundaries.
fn meshBox(ray: Ray, inverse: vec3<f32>, parallel: vec3<bool>, index: u32, tmax: f32) -> f32 {
    return meshSlab(ray,inverse,parallel,16u+index*8u,tmax);
}
fn meshSlab(ray:Ray,inverse:vec3<f32>,parallel:vec3<bool>,node:u32,tmax:f32)->f32 {
    return meshBounds(ray,inverse,parallel,meshVec3(node),meshVec3(node+4u),tmax);
}
fn meshBounds(ray:Ray,inverse:vec3<f32>,parallel:vec3<bool>,lo:vec3<f32>,hi:vec3<f32>,tmax:f32)->f32 {
    if any(parallel & ((ray.origin < lo) | (ray.origin > hi))) {return MAX_T;}
    let a=(lo-ray.origin)*inverse; let b=(hi-ray.origin)*inverse;
    let lower=select(min(a,b),vec3(-MAX_T),parallel);
    let upper=select(max(a,b),vec3(MAX_T),parallel);
    let near=max(MIN_T,max(lower.x,max(lower.y,lower.z)));
    let far=min(tmax,min(upper.x,min(upper.y,upper.z)));
    return select(MAX_T,near,near<=far);
}
struct MeshCandidate { t: f32, u: f32, v: f32, index: u32 }
fn meshCandidate(ray: Ray, index: u32, tmax: f32, outHit: ptr<function, MeshCandidate>) -> bool {
    let accel=meshData[8u]+index*12u;
    let a=meshVec3(accel); let e1=meshVec3(accel+4u); let e2=meshVec3(accel+8u);
    let crossD=cross(ray.direction,e2); let determinant=dot(e1,crossD);
    if abs(determinant)<=meshFloat(accel+3u)*length(ray.direction) {return false;}
    let flags=meshData[accel+11u];
    if (flags&1u)==0u && determinant<0. {return false;}
    let inv=1./determinant;let delta=ray.origin-a;
    let u=dot(delta,crossD)*inv;
    if u<0. || u>1. {return false;}
    let q=cross(delta,e1);let v=dot(ray.direction,q)*inv;
    if v<0. || u+v>1. {return false;}
    let t=dot(e2,q)*inv;
    if t<=MIN_T || t>=tmax {return false;}
    var knownOpaque=false;
    if SOFTWARE_OPACITY && (flags&2u)!=0u {
        let cell=min(u32(u*4.),3u)+4u*min(u32(v*4.),3u);
        let state=(meshData[meshData[10u]+index]>>(2u*cell))&3u;
        if state==2u {return false;}
        knownOpaque=state==1u;
    }
    if !knownOpaque && (flags&2u)!=0u && OPACITY_APPROX && pathDiffuse {
        let coverage=meshFloat(meshData[13u]+meshData[accel+7u]);
        let threshold=(1.-sqrt(coverage))/3.;
        if coverage<=0. || min(1.-u-v,min(u,v))<threshold{return false;}
    } else if !knownOpaque && (flags&2u)!=0u && (!ALPHA_CAP || alphaQueries<4u) {
        if ALPHA_CAP {alphaQueries+=1u;}
        let tri=meshData[2u]+index*4u;
        let va=vertexOffset(meshData[tri]);
        let vb=vertexOffset(meshData[tri+1u]);
        let vc=vertexOffset(meshData[tri+2u]);
        let material=meshMaterialOffset(meshData[accel+7u]);
        let weights=vec3(1.-u-v,u,v);
        let uv0=weights.x*vertexUV0(va)+weights.y*vertexUV0(vb)+weights.z*vertexUV0(vc);
        let uv1=weights.x*vertexUV1(va)+weights.y*vertexUV1(vb)+weights.z*vertexUV1(vc);
        let alpha=meshFloat(material+3u)*meshTexture(material+16u,uv0,uv1,false).a;
        if alpha<meshFloat(material+10u) {return false;}
    }
    *outHit=MeshCandidate(t,u,v,index);
    return true;
}

// Resolve attributes and sample the normal map once, after closest-hit traversal.
fn meshResolve(ray: Ray, candidate: MeshCandidate) -> Intersection {
    let tri=meshData[2u]+candidate.index*4u;
    let va=vertexOffset(meshData[tri]);
    let vb=vertexOffset(meshData[tri+1u]);
    let vc=vertexOffset(meshData[tri+2u]);
    let a=meshVec3(va); let e1=meshVec3(vb)-a; let e2=meshVec3(vc)-a;
    let materialIdx=meshData[tri+3u];let material=meshMaterialOffset(materialIdx);
    let weights=vec3(1.-candidate.u-candidate.v,candidate.u,candidate.v);
    let uv0=weights.x*vertexUV0(va)+weights.y*vertexUV0(vb)+weights.z*vertexUV0(vc);
    let uv1=weights.x*vertexUV1(va)+weights.y*vertexUV1(vb)+weights.z*vertexUV1(vc);
    if RAY_CONES {
        coneWidth+=coneSpread*candidate.t*length(ray.direction);
        let areaVector=cross(e1,e2);let inverseArea=1./max(dot(areaVector,areaVector),1e-20);
        let dual1=cross(e2,areaVector)*inverseArea;let dual2=cross(areaVector,e1)*inverseArea;
        let a0=vertexUV0(va);let b0=vertexUV0(vb)-a0;let c0=vertexUV0(vc)-a0;
        let a1=vertexUV1(va);let b1=vertexUV1(vb)-a1;let c1=vertexUV1(vc)-a1;
        footprintUV0=coneWidth*vec2(length(b0.x*dual1+c0.x*dual2),length(b0.y*dual1+c0.y*dual2));
        footprintUV1=coneWidth*vec2(length(b1.x*dual1+c1.x*dual2),length(b1.y*dual1+c1.y*dual2));
    }
    if SKIP_FINAL_SCATTER && pathBounce+1u==samplingParams.numBounces {
        return Intersection(ray.origin+candidate.t*ray.direction,vec3(0.),uv0.x,uv0.y,candidate.t,materialIdx,0xffffffffu,uv1,vec3(0.));
    }
    var geo=normalize(cross(e1,e2));
    var n=weights.x*vertexNormal(va)+weights.y*vertexNormal(vb)+weights.z*vertexNormal(vc);
    if dot(n,n)<1e-12 {n=geo;} else {n=normalize(n);}
    if dot(n,geo)<0. {n=-n;}
    if meshData[material+40u]!=0xffffffffu {
        let slot=material+40u;
        let tangent=weights.x*vertexTangent(va)+weights.y*vertexTangent(vb)+weights.z*vertexTangent(vc);
        var frame=pixarOnb(n);
        if abs(tangent.w)>0.5 && meshData[slot+1u]==0u && meshFloat(slot+6u)==0. && all(meshVec2(slot+4u)==vec2(1.)) {
            let direction=tangent.xyz-n*dot(n,tangent.xyz);
            if dot(direction,direction)>1e-12 {
                let tx=normalize(direction);frame=mat3x3(tx,cross(n,tx)*sign(tangent.w),n);
            }
        } else {
            let ta=meshUV(slot,vertexUV0(va),vertexUV1(va));
            let tb=meshUV(slot,vertexUV0(vb),vertexUV1(vb))-ta;
            let tc=meshUV(slot,vertexUV0(vc),vertexUV1(vc))-ta;
            let det=tb.x*tc.y-tb.y*tc.x;
            if abs(det)>1e-10 {
                let tx=(e1*tc.y-e2*tb.y)/det;
                let ty=(e2*tb.x-e1*tc.x)/det;
                let ortho=tx-n*dot(n,tx);
                if dot(ortho,ortho)>1e-12 {
                    let tangentN=normalize(ortho);
                    frame=mat3x3(tangentN,cross(n,tangentN)*sign(dot(cross(n,tangentN),ty)),n);
                }
            }
        }
        var normal=meshTexture(slot,uv0,uv1,false).xyz*2.-vec3(1.);
        normal=vec3(normal.xy*meshFloat(material+9u),normal.z);
        let mapped=frame*normal;
        if dot(mapped,mapped)>1e-12 && dot(mapped,geo)>0. {n=normalize(mapped);}
    }
    if dot(geo,ray.direction)>0. {geo=-geo;n=-n;}
    return Intersection(ray.origin+candidate.t*ray.direction,n,uv0.x,uv0.y,candidate.t,materialIdx,0xffffffffu,uv1,geo);
}

// Reference/diagnostic callers retain the full intersection interface.
fn meshTriangle(ray: Ray, index: u32, tmax: f32, outHit: ptr<function, Intersection>) -> bool {
    var candidate=MeshCandidate();
    if !meshCandidate(ray,index,tmax,&candidate) {return false;}
    *outHit=meshResolve(ray,candidate);
    return true;
}
fn meshIntersection(ray: Ray, outHit: ptr<function, Intersection>, tmax: f32) -> bool {
    if WIDE_BVH>2u{return meshWideQuery(ray,outHit,tmax,false);}
    if meshData[0u]==0u {return false;}
    var closest=tmax;var found=false;var best=MeshCandidate();
    let parallel=abs(ray.direction)<vec3(1e-20);
    let inverse=vec3(1.)/select(ray.direction,vec3(1.),parallel);
    var stack:array<u32,64>;var size=0u;var index=0u;var needsBox=true;
    loop {
        if !needsBox || meshBox(ray,inverse,parallel,index,closest)<closest {
            let node=16u+index*8u;let first=meshData[node+3u];let count=meshData[node+7u];
            if count!=0u {
                for(var i=0u;i<count;i+=1u) {
                    var hit=MeshCandidate();
                    if meshCandidate(ray,first+i,closest,&hit) {closest=hit.t;best=hit;found=true;}
                }
            } else {
                let left=index+1u;let right=first;
                let a=meshBox(ray,inverse,parallel,left,closest);let b=meshBox(ray,inverse,parallel,right,closest);
                if min(a,b)<closest {
                    if max(a,b)<closest {stack[size]=select(left,right,a<=b);size+=1u;}
                    index=select(right,left,a<=b);needsBox=false;continue;
                }
            }
        }
        if size==0u {break;}
        size-=1u;index=stack[size];needsBox=true;
    }
    if found {*outHit=meshResolve(ray,best);}
    return found;
}

// glTF metallic/roughness: Lambert diffuse + GGX/Smith microfacets, Schlick Fresnel.
// Sample the diffuse/specular mixture and divide by the complete mixture PDF.
fn scatterPbr(inRay: Ray, hit: Intersection, rng: ptr<function,u32>) -> Scatter {
    if GGX_VNDF || GGX_BOUNDED {return samplePbr(pbrSurface(inRay,hit),hit,rng).scatter;}
    let m=meshMaterialOffset(hit.materialIdx); let uv0=vec2(hit.u,hit.v);
    let base=(meshVec4(m)*meshTexture(m+16u,uv0,hit.uv1,true)).rgb;
    let mr=meshTexture(m+24u,uv0,hit.uv1,false);
    let metal=clamp(meshFloat(m+7u)*mr.b,0.,1.);
    let rough=clamp(meshFloat(m+8u)*mr.g,0.045,1.);
    let alpha=rough*rough;let alpha2=alpha*alpha;
    let view=normalize(-inRay.direction);let nv=max(dot(hit.n,view),1e-6);
    let f0=mix(vec3(0.04),base,metal);
    let specChance=clamp(max(f0.r,max(f0.g,f0.b)),0.25,0.9);
    var wi=vec3(0.);
    let specular=rngNextFloat(rng)<specChance;
    if OPACITY_APPROX {pathDiffuse=!specular;}
    if specular {
        let r1=rngNextFloat(rng); let r2=min(rngNextFloat(rng),0.999999);
        let cosTheta=sqrt((1.-r2)/(1.+(alpha2-1.)*r2));
        let sinTheta=sqrt(max(0.,1.-cosTheta*cosTheta));let phi=2.*PI*r1;
        let halfVector=pixarOnb(hit.n)*vec3(cos(phi)*sinTheta,sin(phi)*sinTheta,cosTheta);
        wi=reflect(-view,halfVector);
    } else {wi=sampleLambertian(hit,rng);}
    let nl=dot(hit.n,wi);
    if nl<=0. || dot(hit.geometricNormal,wi)<=0. {return Scatter(Ray(hit.p,wi),vec3(0.));}
    let h=normalize(wi+view);let nh=max(dot(hit.n,h),0.);let vh=max(dot(view,h),1e-6);
    let denom=nh*nh*(alpha2-1.)+1.;
    let distribution=alpha2/(PI*denom*denom);
    let gv=2.*nv/(nv+sqrt(alpha2+(1.-alpha2)*nv*nv));
    let gl=2.*nl/(nl+sqrt(alpha2+(1.-alpha2)*nl*nl));
    let fresnel=f0+(vec3(1.)-f0)*pow(1.-vh,5.);
    let brdf=(vec3(1.)-fresnel)*(1.-metal)*base*FRAC_1_PI+fresnel*distribution*gv*gl/(4.*nv*nl);
    let pdf=specChance*distribution*nh/(4.*vh)+(1.-specChance)*nl*FRAC_1_PI;
    return Scatter(Ray(hit.p+hit.geometricNormal*0.0001,wi),brdf*nl/max(pdf,1e-12));
}

// Shadow queries stop on the first accepted surface; cutout tests stay in meshCandidate.
fn meshOccluded(ray: Ray) -> bool {
    if WIDE_BVH>2u{var ignored=Intersection();return meshWideQuery(ray,&ignored,MAX_T,true);}
    if meshData[0u]==0u {return false;}
    let parallel=abs(ray.direction)<vec3(1e-20);
    let inverse=vec3(1.)/select(ray.direction,vec3(1.),parallel);
    var stack:array<u32,64>;var size=0u;var index=0u;var needsBox=true;
    loop {
        if !needsBox || meshBox(ray,inverse,parallel,index,MAX_T)<MAX_T {
            let node=16u+index*8u;let first=meshData[node+3u];let count=meshData[node+7u];
            if count!=0u {
                for(var i=0u;i<count;i+=1u) {
                    var hit=MeshCandidate();
                    if meshCandidate(ray,first+i,MAX_T,&hit) {return true;}
                }
            } else {
                let left=index+1u;let right=first;
                let a=meshBox(ray,inverse,parallel,left,MAX_T);let b=meshBox(ray,inverse,parallel,right,MAX_T);
                if min(a,b)<MAX_T {
                    if max(a,b)<MAX_T {stack[size]=select(left,right,a<=b);size+=1u;}
                    index=select(right,left,a<=b);needsBox=false;continue;
                }
            }
        }
        if size==0u {break;}
        size-=1u;index=stack[size];needsBox=true;
    }
    return false;
}

// Loaded once per vertex, shared by environment evaluation and BSDF sampling.
struct PbrSurface {base:vec3<f32>,metal:f32,alpha2:f32,view:vec3<f32>,nv:f32,f0:vec3<f32>,specChance:f32}
struct PbrSample {scatter:Scatter,pdf:f32}
fn pbrSurface(inRay: Ray,hit: Intersection) -> PbrSurface {
    let m=meshMaterialOffset(hit.materialIdx);let uv0=vec2(hit.u,hit.v);
    let base=(meshVec4(m)*meshTexture(m+16u,uv0,hit.uv1,true)).rgb;
    let mr=meshTexture(m+24u,uv0,hit.uv1,false);
    let metal=clamp(meshFloat(m+7u)*mr.b,0.,1.);
    let rough=clamp(meshFloat(m+8u)*mr.g,0.045,1.);
    let alpha=rough*rough;
    let view=normalize(-inRay.direction);
    let f0=mix(vec3(0.04),base,metal);
    return PbrSurface(base,metal,alpha*alpha,view,max(dot(hit.n,view),1e-6),f0,clamp(max(f0.r,max(f0.g,f0.b)),0.25,0.9));
}
fn evaluatePbr(s: PbrSurface,hit: Intersection,wi: vec3<f32>) -> vec4<f32> {
    return evaluatePbrMode(s,hit,wi,GGX_VNDF || GGX_BOUNDED);
}
fn evaluatePbrMode(s: PbrSurface,hit: Intersection,wi: vec3<f32>,visible: bool) -> vec4<f32> {
    let nl=dot(hit.n,wi);
    if nl<=0. || dot(hit.geometricNormal,wi)<=0. {return vec4(0.);}
    let h=normalize(wi+s.view);let nh=max(dot(hit.n,h),0.);let vh=max(dot(s.view,h),1e-6);
    let denom=nh*nh*(s.alpha2-1.)+1.;
    let distribution=s.alpha2/(PI*denom*denom);
    let gv=2.*s.nv/(s.nv+sqrt(s.alpha2+(1.-s.alpha2)*s.nv*s.nv));
    let gl=2.*nl/(nl+sqrt(s.alpha2+(1.-s.alpha2)*nl*nl));
    let fresnel=s.f0+(vec3(1.)-s.f0)*pow(1.-vh,5.);
    let brdf=(vec3(1.)-fresnel)*(1.-s.metal)*s.base*FRAC_1_PI+fresnel*distribution*gv*gl/(4.*s.nv*nl);
    var specularPdf=distribution*nh/(4.*vh);
    if visible && dot(hit.n,s.view)>1e-6 {
        specularPdf=distribution*gv/(4.*s.nv);
        if GGX_BOUNDED {specularPdf=distribution/(2.*(boundedGgxK(s.nv,s.alpha2)*s.nv+sqrt(s.alpha2*(1.-s.nv*s.nv)+s.nv*s.nv)));}
    }
    let pdf=s.specChance*specularPdf+(1.-s.specChance)*nl*FRAC_1_PI;
    return vec4(brdf*nl,pdf);
}
fn samplePbr(s: PbrSurface,hit: Intersection,rng: ptr<function,u32>) -> PbrSample {
    return samplePbrMode(s,hit,rng,GGX_VNDF || GGX_BOUNDED);
}
fn samplePbrMode(s: PbrSurface,hit: Intersection,rng: ptr<function,u32>,visible: bool) -> PbrSample {
    var wi=vec3(0.);
    let specular=rngNextFloat(rng)<s.specChance;
    if OPACITY_APPROX {pathDiffuse=!specular;}
    if specular {
        let r1=rngNextFloat(rng);let r2=min(rngNextFloat(rng),0.999999);
        if visible && dot(hit.n,s.view)>1e-6 {
            let frame=pixarOnb(hit.n);
            let localView=transpose(frame)*s.view;
            let halfVector=frame*sampleVisibleGgx(localView,sqrt(s.alpha2),r1,r2);
            wi=reflect(-s.view,halfVector);
        } else {
            let cosTheta=sqrt((1.-r2)/(1.+(s.alpha2-1.)*r2));
            let sinTheta=sqrt(max(0.,1.-cosTheta*cosTheta));let phi=2.*PI*r1;
            let halfVector=pixarOnb(hit.n)*vec3(cos(phi)*sinTheta,sin(phi)*sinTheta,cosTheta);
            wi=reflect(-s.view,halfVector);
        }
    } else {wi=sampleLambertian(hit,rng);}
    let ev=evaluatePbrMode(s,hit,wi,visible);
    return PbrSample(Scatter(Ray(hit.p+hit.geometricNormal*0.0001,wi),ev.rgb/max(ev.a,1e-12)),ev.a);
}

// Isotropic GGX visible-normal sampling via projected hemisphere sampling.
// Heitz, JCGT 7(4), 2018: https://jcgt.org/published/0007/04/01/
// Below-surface reflection samples are kept as zero contribution, never retried.
fn sampleVisibleGgx(view: vec3<f32>,alpha: f32,u: f32,v: f32) -> vec3<f32> {
    let stretched=normalize(vec3(alpha*view.xy,view.z));
    if GGX_BOUNDED {
        let z=1.-v*(1.+boundedGgxK(view.z,alpha*alpha)*stretched.z);
        let r=sqrt(max(0.,1.-z*z));let phi=2.*PI*u;
        let normal=stretched+vec3(r*cos(phi),r*sin(phi),z);
        return normalize(vec3(alpha*normal.xy,normal.z));
    }
    let tangentLength=dot(stretched.xy,stretched.xy);
    var tangent=vec3(1.,0.,0.);
    if tangentLength>0. {tangent=vec3(-stretched.y,stretched.x,0.)/sqrt(tangentLength);}
    let bitangent=cross(stretched,tangent);
    let radius=sqrt(u);let angle=2.*PI*v;
    let diskX=radius*cos(angle);
    let blend=0.5*(1.+stretched.z);
    let diskY=(1.-blend)*sqrt(max(0.,1.-diskX*diskX))+blend*radius*sin(angle);
    let projected=diskX*tangent+diskY*bitangent+sqrt(max(0.,1.-diskX*diskX-diskY*diskY))*stretched;
    return normalize(vec3(alpha*projected.xy,max(0.,projected.z)));
}

// Packed shading attributes retain float32 positions for traversal.
fn meshNormal(address:u32)->vec3<f32>{
    if meshData[address]==0x80008000u{return vec3(0.);}
    let p=unpack2x16snorm(meshData[address]);
    var n=vec3(p,1.-abs(p.x)-abs(p.y));
    if n.z<0. {n=vec3((vec2(1.)-abs(n.yx))*select(vec2(-1.),vec2(1.),n.xy>=vec2(0.)),n.z);}
    return normalize(n);
}
fn vertexOffset(index:u32)->u32 {return meshData[3u]+index*select(16u,8u,PACKED_SHADING);}
fn vertexNormal(address:u32)->vec3<f32>{if PACKED_SHADING{return meshNormal(address+3u);}return meshVec3(address+4u);}
fn vertexUV0(address:u32)->vec2<f32>{if PACKED_SHADING{return unpack2x16float(meshData[address+6u]);}return meshVec2(address+8u);}
fn vertexUV1(address:u32)->vec2<f32>{if PACKED_SHADING{return unpack2x16float(meshData[address+7u]);}return meshVec2(address+10u);}
fn vertexTangent(address:u32)->vec4<f32>{if PACKED_SHADING{return vec4(meshNormal(address+4u),meshFloat(address+5u));}return meshVec4(address+12u);}

fn meshBruteQuery(ray:Ray,outHit:ptr<function,Intersection>,limit:f32,anyHit:bool)->bool{
    var closest=limit;var found=false;var best=MeshCandidate();
    for(var i=0u;i<meshData[1u];i+=1u){var hit=MeshCandidate();if meshCandidate(ray,i,closest,&hit){if anyHit{return true;}found=true;closest=hit.t;best=hit;}}
    if found{*outHit=meshResolve(ray,best);}return found;
}
fn meshWideQuery(ray:Ray,outHit:ptr<function,Intersection>,limit:f32,anyHit:bool)->bool{
    if meshData[0u]==0u{return false;}
    let parallel=abs(ray.direction)<vec3(1e-20);let inverse=vec3(1.)/select(ray.direction,vec3(1.),parallel);
    var stack:array<vec2<u32>,64>;var size=0u;var group=0u;var expand=true;
    var closest=limit;var best=MeshCandidate();var found=false;
    loop{
        if expand{
            var children:array<vec2<u32>,8>;var count=0u;
            for(var lane=0u;lane<WIDE_BVH;lane+=1u){
                let record=meshData[9u]+(group*WIDE_BVH+lane)*select(8u,4u,QUANTIZED_BVH);
                if wideFirst(record)==select(0xffffffffu,0xffffffu,QUANTIZED_BVH){continue;}
                let near=wideSlab(ray,inverse,parallel,record,closest);
                if near>closest || near==MAX_T{continue;}
                var j=count;
                while j>0u && bitcast<f32>(children[j-1u].y)<near{children[j]=children[j-1u];j-=1u;}
                children[j]=vec2(record,bitcast<u32>(near));count+=1u;
            }
            // Rare overflow preserves correctness through a complete fallback query.
            if size+count>64u{return meshBruteQuery(ray,outHit,limit,anyHit);}
            for(var i=0u;i<count;i+=1u){stack[size]=children[i];size+=1u;}
        }
        expand=false;
        if size==0u{break;}
        size-=1u;let pending=stack[size];
        if bitcast<f32>(pending.y)>closest{continue;}
        let first=wideFirst(pending.x);let count=wideCount(pending.x);
        if count==0u{group=first;expand=true;continue;}
        for(var i=0u;i<count;i+=1u){
            var hit=MeshCandidate();
            let limit=select(closest,bitcast<f32>(bitcast<u32>(closest)+1u),found);
            if meshCandidate(ray,first+i,limit,&hit){
                if anyHit{return true;}
                if !found || hit.t<closest || (hit.t==closest && hit.index<best.index){best=hit;closest=hit.t;found=true;}
            }
        }
    }
    if found{*outHit=meshResolve(ray,best);}return found;
}

fn boundedGgxK(nv:f32,alpha2:f32)->f32{let s=1.+sqrt(max(0.,1.-nv*nv));return (1.-alpha2)*s*s/(s*s+alpha2*nv*nv);}

fn wideFirst(record:u32)->u32{if QUANTIZED_BVH{return meshData[record+3u]&0xffffffu;}return meshData[record+3u];}
fn wideCount(record:u32)->u32{if QUANTIZED_BVH{return meshData[record+3u]>>24u;}return meshData[record+7u];}
fn wideSlab(ray:Ray,inverse:vec3<f32>,parallel:vec3<bool>,record:u32,limit:f32)->f32{
    if !QUANTIZED_BVH{return meshSlab(ray,inverse,parallel,record,limit);}
    let a=unpack2x16float(meshData[record]);let b=unpack2x16float(meshData[record+1u]);let c=unpack2x16float(meshData[record+2u]);
    return meshBounds(ray,inverse,parallel,vec3(a,b.x),vec3(b.y,c),limit);
}
