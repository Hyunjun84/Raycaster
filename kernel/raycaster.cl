#ifndef M_PI
# define M_PI 3.141592653589793115998
#endif

#ifndef M_PI_F
# define M_PI_F 3.14159274101257f
#endif

const sampler_t sp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__inline float eval(float3 p, __read_only image3d_t vol);
__inline float3 eval_g(float3 p, __read_only image3d_t vol);
__inline float8 eval_H(float3 p, __read_only image3d_t vol);

__inline float4 MDotV(float16 m, float4 v) 
{
    return (float4)(dot(m.s0123, v),dot(m.s4567, v), dot(m.s89ab, v), dot(m.scdef, v));
}

__kernel void genRay(__global float8* ray, float16 M, float4 data_ratio)
{
	int2 idx = (int2)(get_global_id(0), get_global_id(1));
	int2 sz  = (int2)(get_global_size(0), get_global_size(1));

	int id = sz.x*idx.y+idx.x;

	float4 b = (float4)(convert_float2(idx*2)/convert_float2(sz-1)-1.0f, -1.0f, 1.0f); 
	float4 e = (float4)(b.xy, 1.0f, 1.0f);

	b = MDotV(M, b);
    e = MDotV(M, e);

    b = b/b.w;
    e = e/e.w;

    float4 d = (float4)(normalize(e.xyz-b.xyz), 0);

    float2 hit_yz = (float2)(-data_ratio.x-b.x, data_ratio.x-b.x)/d.x;
    float2 hit_zx = (float2)(-data_ratio.y-b.y, data_ratio.y-b.y)/d.y; 
    float2 hit_xy = (float2)(-data_ratio.z-b.z, data_ratio.z-b.z)/d.z;

    hit_yz = (float2)(min(hit_yz.x, hit_yz.y), max(hit_yz.x, hit_yz.y));
    hit_zx = (float2)(min(hit_zx.x, hit_zx.y), max(hit_zx.x, hit_zx.y));
    hit_xy = (float2)(min(hit_xy.x, hit_xy.y), max(hit_xy.x, hit_xy.y));

    float2 bound = (float2)(
        max(max(hit_yz.x, hit_zx.x), hit_xy.x),
        min(min(hit_yz.y, hit_zx.y), hit_xy.y));
   
    if(any(isnan(bound)) || any(isinf(bound)) || (bound.x>bound.y))
        bound = (float2)(0);

    ray[id] = (float8)(b.xyz+d.xyz*bound.x, b.w, b.xyz+d.xyz*bound.y, e.w);
}

__kernel void raycast(__write_only image2d_t Position, __read_only image3d_t vol, __global float8* Rays, float4 scale, float4 dim, float level)
{
    int2 id = (int2)(get_global_id(0), get_global_id(1));
    int2 sz  = (int2)(get_global_size(0), get_global_size(1));

	float8 ray = Rays[sz.x*id.y+id.x];

    // Normalized domain -> texture coordinates
	float3 p = (ray.s012/scale.xyz*0.5f+0.5f)*dim.xyz-0.5f;
    float3 e = (ray.s456/scale.xyz*0.5f+0.5f)*dim.xyz-0.5f; // [-0.5 ... dim-0.5]^3
	float3 p_prev;

	float ray_step = 0.4f;
    float max_ray_len = distance(p,e);
    float voxel = eval(p, vol);
    float voxel_prev = voxel;
    
    float orientation = 2.f*convert_float(voxel < level)-1.f;     // equivalent to (voxel<level?1:-1)
    float3 dir = normalize(e-p)*ray_step;

    int max_iter = min(100000, convert_int(max_ray_len/ray_step));//convert_int(ray.w*fdim.z/dir.z));
    float4 val = (float4)(0);

    int i=0;

    for(i=0; i<max_iter; i++) {
    	p = p+dir.xyz;
    	voxel = eval(p, vol);
		if(orientation*voxel > orientation*level) {
            // One step of Regula Falsi
            if(fabs(voxel-voxel_prev) > 1E-4) 
                p = (p*(voxel_prev-level) - p_prev*(voxel-level))/(voxel_prev-voxel);
            // store normalized coordinates( [0...1]^3*scale)
            val = (float4)( ((p+0.5f)/dim.xyz)*scale.xyz, orientation);
            break;
        }
        voxel_prev=voxel;
        p_prev=p;
    }
    
    write_imagef(Position, id, val);
}

__kernel void evalGradient(__write_only image2d_t Gradient, __read_only image2d_t Position, __read_only image3d_t vol, float4 dim, float4 scale)
{
    int2 id = (int2)(get_global_id(0), get_global_id(1));
    float4 p = read_imagef(Position, sp, id);
    
    // convert texture coordinates( [-0.5...(dim-0.5)]
    p.xyz = (p.xyz/scale.xyz)*dim.xyz-0.5f;

    float3 grad = (float3)(0);
    if(p.w!=0)  grad = eval_g(p.xyz, vol);
    
    write_imagef(Gradient, id, (float4)((grad/dim.xyz)*scale.xyz,1));
}


__kernel void evalFiniteGradient(__write_only image2d_t Gradient, __read_only image2d_t Position, __read_only image3d_t vol, float4 dim, float4 scale)
{
    int2 id = (int2)(get_global_id(0), get_global_id(1));
    float4 p = read_imagef(Position, sp, id);
    // convert texture coordinates( [-0.5...(dim-0.5)]
    p.xyz = (p.xyz/scale.xyz)*dim.xyz-0.5f;

    float3 grad = (float3)(0);
    if(p.w!=0){
        float delta = 0.1f;
        float d011 = eval(p.xyz-(float3)(delta, 0, 0), vol);
        float d211 = eval(p.xyz+(float3)(delta, 0, 0), vol);
        float d101 = eval(p.xyz-(float3)(0, delta, 0), vol);
        float d121 = eval(p.xyz+(float3)(0, delta, 0), vol);
        float d110 = eval(p.xyz-(float3)(0, 0, delta), vol);
        float d112 = eval(p.xyz+(float3)(0, 0, delta), vol);
        grad = (float3)(d211-d011, d121-d101, d112-d110)/(2*delta);
    }

    write_imagef(Gradient, id, (float4)(grad/dim.xyz*scale.xyz,1));
}

__kernel void evalHessian(__write_only image2d_t Hessian1, __write_only image2d_t Hessian2, __read_only image2d_t Position, __read_only image3d_t vol, float4 scale, float4 dim)
{
    int2 id = (int2)(get_global_id(0), get_global_id(1));
    float4 p = read_imagef(Position, sp, id);
    // convert texture coordinates( [-0.5...(dim-0.5)]
    p.xyz = (p.xyz/scale.xyz)*dim.xyz-0.5f;


    float8 H = (float8)(0);
    if(p.w!=0) H = eval_H(p.xyz, vol);

    float4 inv_scale = 1.f/(dim*scale);

    write_imagef(Hessian1, id, H.lo*inv_scale*inv_scale);
    write_imagef(Hessian2, id, H.hi*inv_scale.yzxw*inv_scale.zxyw);
}

__kernel void evalFiniteHessian(__write_only image2d_t Hessian1, __write_only image2d_t Hessian2, __read_only image2d_t Position, __read_only image3d_t vol, float4 scale, float4 dim)
{
    int2 id = (int2)(get_global_id(0), get_global_id(1));
    float4 p = read_imagef(Position, sp, id);
    // convert texture coordinates( [-0.5...(dim-0.5)]
    p.xyz = (p.xyz/scale.xyz)*dim.xyz-0.5f;

    float8 H = (float8)(0);
    if(p.w!=0) {
        float delta = 0.05;
        float f111 = eval(p.xyz, vol);

        float f001 = eval(p.xyz + (float3)(-delta,-delta,     0), vol);
        float f010 = eval(p.xyz + (float3)(-delta,     0,-delta), vol);
        float f011 = eval(p.xyz + (float3)(-delta,     0,     0), vol);
        float f012 = eval(p.xyz + (float3)(-delta,     0, delta), vol);
        float f021 = eval(p.xyz + (float3)(-delta, delta,     0), vol);

        float f100 = eval(p.xyz + (float3)(     0,-delta,-delta), vol);
        float f101 = eval(p.xyz + (float3)(     0,-delta,     0), vol);
        float f102 = eval(p.xyz + (float3)(     0,-delta, delta), vol);
        float f110 = eval(p.xyz + (float3)(     0,     0,-delta), vol);
        float f112 = eval(p.xyz + (float3)(     0,     0, delta), vol);
        float f120 = eval(p.xyz + (float3)(     0, delta,-delta), vol);
        float f121 = eval(p.xyz + (float3)(     0, delta,     0), vol);
        float f122 = eval(p.xyz + (float3)(     0, delta, delta), vol);

        float f201 = eval(p.xyz + (float3)( delta,-delta,     0), vol);
        float f210 = eval(p.xyz + (float3)( delta,     0,-delta), vol);
        float f211 = eval(p.xyz + (float3)( delta,     0,     0), vol);
        float f212 = eval(p.xyz + (float3)( delta,     0, delta), vol);
        float f221 = eval(p.xyz + (float3)( delta, delta,     0), vol);

        float _1_over_delta_square = 1.0f/(delta*delta);

        H.lo.xyz = (float3)((f211 - 2.0f*f111 + f011),
                            (f121 - 2.0f*f111 + f101),
                            (f112 - 2.0f*f111 + f110))*_1_over_delta_square;

        H.hi.xyz = 0.25*(float3)((f122 - f120 - f102 + f100),
                                 (f212 - f210 - f012 + f010),
                                 (f221 - f201 - f021 + f001))*_1_over_delta_square;
    }

    write_imagef(Hessian1, id, H.lo/dim*scale*scale);
    write_imagef(Hessian2, id, H.hi/dim*scale.yzxw*scale.zxyw);
}
