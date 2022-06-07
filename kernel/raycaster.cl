const sampler_t sp = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
const sampler_t sp2 = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
const sampler_t sp3 = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

#define SCALE_F 0.0625

__inline float4 MDotV(float16 m, float4 v) 
{
	return (float4)(dot(m.s0123, v),dot(m.s4567, v), dot(m.s89ab, v), dot(m.scdef, v));
}

__kernel void genRay(__global float8* ray, float16 MVP)
{
	int2 idx = (int2)(get_global_id(0), get_global_id(1));
	int2 sz  = (int2)(get_global_size(0), get_global_size(1));

	int id = sz.x*idx.y+idx.x;

	float4 b = (float4)(convert_float2(idx*2)/convert_float2(sz-1)-1.0, -100.0, 1.0); //[-1...1]^3
	float4 e = (float4)(b.xy, 100.0, 1.0);

	b = MDotV(MVP, b);
    e = MDotV(MVP, e);

    float4 d = normalize(e-b);

    float2 hit_yz = (float2)(-1-b.x, 1-b.x)/d.x;
    float2 hit_zx = (float2)(-1-b.y, 1-b.y)/d.y;
    float2 hit_xy = (float2)(-1-b.z, 1-b.z)/d.z;

    hit_yz = (float2)(min(hit_yz.x, hit_yz.y), max(hit_yz.x, hit_yz.y));
    hit_zx = (float2)(min(hit_zx.x, hit_zx.y), max(hit_zx.x, hit_zx.y));
    hit_xy = (float2)(min(hit_xy.x, hit_xy.y), max(hit_xy.x, hit_xy.y));

    float2 bound =  (float2)(
        max(max(hit_yz.x, hit_zx.x), hit_xy.x),
        min(min(hit_yz.y, hit_zx.y), hit_xy.y));
   
    if(any(isnan(bound)) || any(isinf(bound)) )
        bound = (float2)(0);

    ray[id] = (float8)(b.xyz+d.xyz*bound.x, 1, b.xyz+d.xyz*bound.y, 1);
}


__inline float eval(float3 p_in, __read_only image3d_t vol)
{
	// Find origin
	int3 org = convert_int3(round(p_in));
    if( (convert_int(org.x+org.y+org.z)&0x01) != 0 ) {
        float3 l = p_in-org+1e-7;
        float3 a  = fabs(l);
        if(a.x > a.y) {
            if(a.x > a.z)   org.x += convert_int(sign(l.x));
            else            org.z += convert_int(sign(l.z));
        } else {
            if(a.y > a.z)   org.y += convert_int(sign(l.y));
            else            org.z += convert_int(sign(l.z));
        }
    }
    float3 p_local = p_in - convert_float3(org);    // local coordinates

    // computes the membership against the six knot planes intersecting the unit cube centered at the local origin
    int d[6] = {  
    	convert_int(p_local.z>=p_local.y),
		convert_int(p_local.z>=p_local.x),
		convert_int(p_local.y>=p_local.x),
		convert_int(p_local.x>=-p_local.y),
		convert_int(p_local.x>=-p_local.z),
		convert_int(p_local.y>=-p_local.z)
	};

    // type_R: the `reflection transformation' which is one of four `even reflections'
    // The reflection matrix R for each type:
    // (0,0,0): [ 1, 0, 0] (0,1,1): [ 1, 0, 0] (1,0,1): [-1, 0, 0] (1,1,0): [-1, 0, 0]
    //          [ 0, 1, 0]          [ 0,-1, 0]          [ 0, 1, 0]          [ 0,-1, 0]
    //          [ 0, 0, 1]          [ 0, 0,-1]          [ 0, 0,-1]          [ 0, 0, 1]
    int3 type_R = 	(1-d[1])*(1-d[2])*(1-d[5])*(int3)(0,1,1) + 
    				(1-d[0])*d[2]*(1-d[4])*(int3)(1,0,1) +
    				d[0]*d[1]*(1-d[3])*(int3)(1,1,0);
    
    // Transform p_local into the `reference left coset' (Fig 2(a)) hit_zx the reflection computed above.
    // Same as R^-1*p_local (R is one of the reflection matrices defined above)
    // Note that R^{-1}=R since R is symmetric & orthogonal.
    float3  p_ref_R = p_local * convert_float3(1-2*type_R);   

    // Compute the membership against the three knot planes intersecting the piece in Fig 2(a).
    // Three knot planes with their normals (-1,1,0), (-1,0,1), and (0,-1,1), respectively.
    // The input (p_ref_R) belong to one of the six tetrahedra in Fig 2(a)
    // and each piece corresponds to one of the six permutation matrices P below.
    int3 type_P = (int3)(convert_int(p_ref_R.y>=p_ref_R.x),    
                         convert_int(p_ref_R.z>=p_ref_R.x), 
                         convert_int(p_ref_R.z>=p_ref_R.y));

    // serialize type_R 
    // (0,0,0)--> 0, (0,1,1)--> 1, (1,0,1)--> 2, (1,1,0)--> 3
    //int idx_R = 2*type_R.x + type_R.y;

    // serialize type_P
    // (0,0,0)--> 0, (0,0,1)--> 1, (1,0,0)--> 2, (0,1,1)--> 3, (1,1,0)--> 4, (1,1,1)--> 5
    int idx_P = 2*(type_P.x+type_P.y)+type_P.z;

    // store type_R and type_P in vector form
    //vecR = int[](int(idx_R==0),int(idx_R==1),int(idx_R==2),int(idx_R==3));
    //vecP = int[](int(idx_P==0),int(idx_P==1),int(idx_P==2),int(idx_P==3),int(idx_P==4),int(idx_P==5));

    int3 vecP1 = (int3)(idx_P==0, idx_P==4, idx_P==3);
    int3 vecP2 = (int3)(idx_P==1, idx_P==2, idx_P==5);
    int3 vecPx = vecP1+vecP2;
    int3 vecPy = vecP1.zxy+vecP2.yzx;
    int3 vecPz = vecP1.yzx+vecP2.zxy;
    
    // Compute the permutation matrix P from type_P.
    // (0,0,0):[1,0,0] (0,0,1):[1,0,0] (1,0,0):[0,1,0] (0,1,1):[0,0,1] (1,1,0):[0,1,0] (1,1,1):[0,0,1]
    //         [0,1,0]         [0,0,1]         [1,0,0]         [1,0,0]         [0,0,1]         [0,1,0]
    //         [0,0,1]         [0,1,0]         [0,0,1]         [0,1,0]         [1,0,0]         [1,0,0]
    // For p_ref_R in one of the six tetrahedral pieces, P^{-1}*p_ref_R is inside the reference tetrahedron.
    // Note that mat3 is in column-major format.
    


	// Transform p_ref_R into the `reference tetrahedron' hit_zx multiplying P.
    float4 p_ref = (float4)(p_ref_R,1);

    if(type_P.y) p_ref.xz = p_ref.zx;
    if(type_P.x) p_ref.xy = p_ref.yx;
    if(type_P.z) p_ref.yz = p_ref.zy;
    if(type_P.x && type_P.y && type_P.z) p_ref.xyz = p_ref.zxy;
    
    // Compute the barycentric coordinates.
    float4 u1 = (float4)(1.0f-p_ref.x-p_ref.y, 
						p_ref.y+p_ref.z, 
						p_ref.x-p_ref.y, 
					    p_ref.y-p_ref.z);
    // Fetch coefficients

    int3 r = 1-2*type_R;
    int3 dirx = r*vecPx;
    int3 diry = r*vecPy;
    int3 dirz = r*vecPz;

    // We re-ordered the fetches such that
    // the displacement of adjacent fetches are axis-aligned.
    // Therefore, we can move to the next coefficient with low computational overhead.
    float c0 = read_imagef(vol, sp2, (int4)(org,1)).r;
    float c1 = read_imagef(vol, sp2, (int4)(org+2*dirx,1)).r;
    float c2 = read_imagef(vol, sp2, (int4)(org+diry-dirz,1)).r;
    float c3 = read_imagef(vol, sp2, (int4)(org+diry+dirz,1)).r;
    float c4 = read_imagef(vol, sp2, (int4)(org+dirx-dirz,1)).r;
    float c5 = read_imagef(vol, sp2, (int4)(org+dirx+dirz,1)).r;
    float c6 = read_imagef(vol, sp2, (int4)(org+dirx-diry,1)).r;
    float c7 = read_imagef(vol, sp2, (int4)(org+dirx+diry,1)).r;


    // Evaluation

    float val = 0;
    float4 u2 = u1*u1;
    float4 u3 = u2*u1;
    
    float c07 = c0+c7;
    float c16 = c1+c6;
    float c0247 = c07+c2+c4;
    float c0357 = c07+c3+c5;
    float c4567 = c4+c5+c6+c7;

    val += u2.x*(u1.y*(8*c0+c0357) +
                 u1.z*(8*c0+c4567) +
                 u1.w*(8*c0+c0247));

    val += u1.x*(2*(u2.y*(2*c0+c0357) +
                    u2.z*(2*c0+c4567) +
                    u2.w*(2*c0+c0247)) +
                u1.z*(u1.w*(4*(c4+c7) + (12*c0 + c4567))+
                      u1.y*(4*(c5+c7) + (12*c0 + c4567)))+
                u1.y*u1.w*(4*c7+c0247+(12*c0 + c0357)));
    
    val += u3.y*c0357 + 
           u3.w*c0247 +
           u1.y*u1.w*(u1.y*(4*c07+c0357) +
                      u1.w*(4*c07+c0247)) + 
           u2.z*(u1.y*(3*(c0357-c3) + (c16+c4))+
                 u1.w*(3*(c0247-c2) + (c16+c5)));

    val += 4*(u1.z*(u2.y*(c0357-c3)+u2.w*(c0247-c2)) + (u3.x*c0));
    val += u1.y*u1.z*u1.w*(2*(c4+c5) + (9*c07+c16));
    val *= 6;
    val += 4*u3.z*(c0+c1+c4567);
    
    
    return val*SCALE_F;
}


__kernel void raycast(__write_only image2d_t Position, __read_only image3d_t vol, __global float8* Rays, float4 scale, int4 dim, float level, __global int2* buf)
{
    int2 id = (int2)(get_global_id(0), get_global_id(1));
    int2 sz  = (int2)(get_global_size(0), get_global_size(1));

	float8 ray = Rays[sz.x*id.y+id.x];

	float3 fdim = convert_float3(dim.xyz-1);
	float3 p = fdim*(0.5f*ray.s012+0.5f); // [0...fdim]^3
    float3 e = fdim*(0.5f*ray.s456+0.5f); // [0...fdim]^3
	float3 p_prev;

	float step = 0.5f;
    float max_ray_len = distance(p,e);
    float voxel = eval(p, vol);
    float voxel_prev = voxel;
    
    float orientation = 2.f*convert_float(voxel < level)-1.f;     // equivalent to (voxel<level?1:-1)
    float3 dir = normalize(e-p)*step;

    int max_iter = min(100000, convert_int(max_ray_len/step)+1);//convert_int(ray.w*fdim.z/dir.z));
    float4 val = (float4)(0);

    int i=0;

    float inverse_scale = 1.f/(fdim.xyz-1.f);
    for(i=0; i<max_iter; i++) {
    	p = p+dir.xyz;
    	voxel = eval(p, vol);
		if(orientation*voxel > orientation*level) {
            // One step of Regula Falsi
            if(fabs(voxel-voxel_prev) > 1E-4) 
                p = (p*(voxel_prev-level) - p_prev*(voxel-level))/(voxel_prev-voxel);
            val = (float4)((p*inverse_scale), orientation);
            break;
        }
        voxel_prev=voxel;
        p_prev=p;
    }
    
    write_imagef(Position, id, val);
}

__kernel void evalGradient(__write_only image2d_t Gradient, __read_only image3d_t vol, __read_only image2d_t Position, int4 dim)
{
    int2 id = (int2)(get_global_id(0), get_global_id(1));

    float delta = 0.1;
    float4 p = read_imagef(Position, sp2, id)*(float4)(convert_float3(dim.xyz-1), 1);

    if(p.w!=0){
        float d011 = eval(p.xyz-(float3)(delta, 0, 0), vol);
        float d211 = eval(p.xyz+(float3)(delta, 0, 0), vol);
        float d101 = eval(p.xyz-(float3)(0, delta, 0), vol);
        float d121 = eval(p.xyz+(float3)(0, delta, 0), vol);
        float d110 = eval(p.xyz-(float3)(0, 0, delta), vol);
        float d112 = eval(p.xyz+(float3)(0, 0, delta), vol);
        float4 grad = (float4)((float3)(d211-d011, d121-d101, d112-d110)/(2*delta), 1);

        write_imagef(Gradient, id, grad);
    }
}

__kernel void hessian(__global float8* dxx, __read_only image2d_t vol, __global float4* pos)
{
	
}
