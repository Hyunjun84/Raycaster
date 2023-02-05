#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

const sampler_t sp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void applyQuasiInterpolator(__write_only image3d_t vol, __read_only image3d_t org_vol, float4 coef, int4 dim)
{
    int4 id = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 1);
    int4 lid = (int4)(get_local_id(0), get_local_id(1), get_local_id(2), 1);
    
    float val = 0.0f;

    if(any(id.xyz>=dim.xyz)) return;

    // 0th
    val = read_imagef(org_vol, sp, id).x*coef.s0;

    // First neighbor
    val += read_imagef(org_vol, sp, id + (int4)(1,0,0,0)).x*coef.s1;
    val += read_imagef(org_vol, sp, id + (int4)(0,1,0,0)).x*coef.s1;
    val += read_imagef(org_vol, sp, id + (int4)(0,0,1,0)).x*coef.s1;
    val += read_imagef(org_vol, sp, id + (int4)(-1,0,0,0)).x*coef.s1;
    val += read_imagef(org_vol, sp, id + (int4)(0,-1,0,0)).x*coef.s1;
    val += read_imagef(org_vol, sp, id + (int4)(0,0,-1,0)).x*coef.s1;

    // Second neighbor
    val += read_imagef(org_vol, sp, id + (int4)(0,1,1,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(1,0,1,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(1,1,0,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)( 0,-1,1,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(-1, 0,1,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(-1, 1,0,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(0, 1,-1,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(1, 0,-1,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(1,-1, 0,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)( 0,-1,-1,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(-1, 0,-1,0)).x*coef.s2;
    val += read_imagef(org_vol, sp, id + (int4)(-1,-1, 0,0)).x*coef.s2;

    // Third neighbor
    val += read_imagef(org_vol, sp, id + (int4)( 1, 1, 1,0)).x*coef.s3;
    val += read_imagef(org_vol, sp, id + (int4)(-1, 1, 1,0)).x*coef.s3;
    val += read_imagef(org_vol, sp, id + (int4)( 1,-1, 1,0)).x*coef.s3;
    val += read_imagef(org_vol, sp, id + (int4)( 1, 1,-1,0)).x*coef.s3;
    val += read_imagef(org_vol, sp, id + (int4)( 1,-1,-1,0)).x*coef.s3;
    val += read_imagef(org_vol, sp, id + (int4)(-1, 1,-1,0)).x*coef.s3;
    val += read_imagef(org_vol, sp, id + (int4)(-1,-1, 1,0)).x*coef.s3;
    val += read_imagef(org_vol, sp, id + (int4)(-1,-1,-1,0)).x*coef.s3;

    write_imagef(vol, id, val);

}