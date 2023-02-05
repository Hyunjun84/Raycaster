#include "./kernel/fcc_v_common.cl"

#define SCALE_F (0.041666667f)
#define SCALE_G (0.0625f)

#define c0 c[0]
#define c1 c[1]
#define c2 c[2]
#define c3 c[3]
#define c4 c[4]
#define c5 c[5]
#define c6 c[6]
#define c7 c[7]


__inline void fetch_coefficients(float* c, image3d_t vol, int3 org, int3 R, uint4 P)
{
    int3 dirx = R*shuffle((int4)(1,0,0,0), P).xyz;
    int3 diry = R*shuffle((int4)(0,1,0,0), P).xyz;
    int3 dirz = R*shuffle((int4)(0,0,1,0), P).xyz;

    c[0] = read_imagef(vol, sp, (int4)(org,1)).r;
    c[1] = read_imagef(vol, sp, (int4)(org+2*dirx,1)).r;

    c[2] = read_imagef(vol, sp, (int4)(org+diry-dirz,1)).r;
    c[3] = read_imagef(vol, sp, (int4)(org+diry+dirz,1)).r;

    c[4] = read_imagef(vol, sp, (int4)(org+dirx-dirz,1)).r;
    c[5] = read_imagef(vol, sp, (int4)(org+dirx+dirz,1)).r;

    c[6] = read_imagef(vol, sp, (int4)(org+dirx-diry,1)).r;
    c[7] = read_imagef(vol, sp, (int4)(org+dirx+diry,1)).r;
}

__inline float eval(float3 p_in, __read_only image3d_t vol)
{
    // Find origin
    int3 org = find_origin(p_in);
    float3 p_local = p_in - convert_float3(org);    // local coordinates

    int3 R = find_R(p_local);
    p_local = convert_float3(R)*p_local;

    uint8 P = find_P(p_local);
    p_local = shuffle((float4)(p_local, 0.f), P.lo).xyz;

    float4 u1 = to_barycentric(p_local);
    
    float c[8];
    fetch_coefficients(c, vol, org, R, P.hi);

    // Evaluation
    float val = 0;
    float4 u2 = u1*u1;
    float4 u3 = u2*u1;
    
    val += u2.x*(u1.y*(9*c0+c3+c5+c7) +
                 u1.z*(8*c0+c4+c5+c6+c7) +
                 u1.w*(9*c0+c2+c4+c7));

    val += u1.x*(2*(u2.y*(3*c0+c3+c5+c7) +
                    u2.z*(2*c0+c4+c5+c6+c7) +
                    u2.w*(3*c0+c2+c4+c7)) +
                u1.z*(u1.w*(12*c0 + 5*(c4+c7) + c5+c6)+
                      u1.y*(12*c0 + 5*(c5+c7) + c4+c6))+
                u1.y*u1.w*(14*c0+6*c7+c2+c3+c4+c5));
    
    val += u3.y*(c0+c3+c5+c7) + 
           u3.w*(c0+c2+c4+c7) +
           u1.y*u1.w*(u1.y*(5*(c0+c7)+c3+c5) +
                      u1.w*(5*(c0+c7)+c2+c4)) + 
           u2.z*(u1.y*(3*(c0+c5+c7)+c1+c4+c6)+
                 u1.w*(3*(c0+c4+c7)+c1+c5+c6));

    val += 4*(u1.z*(u2.y*(c0+c5+c7)+u2.w*(c0+c4+c7)) + u3.x*c0);
    val += u1.y*u1.z*u1.w*(2*(c4+c5) + (9*(c0+c7)+c1+c6));
    val *= 6;
    val += 4*u3.z*(c0+c1+c4+c5+c6+c7);
    
    
    return val*SCALE_F;
}


__inline float3 eval_g(float3 p_in, __read_only image3d_t vol)
{
    // Find origin
    int3 org = find_origin(p_in);
    float3 p_local = p_in - convert_float3(org);    // local coordinates

    int3 R = find_R(p_local);
    p_local = convert_float3(R)*p_local;

    uint8 P = find_P(p_local);
    p_local = shuffle((float4)(p_local, 0), P.lo).xyz;

    float4 u1 = to_barycentric(p_local);
    
    float c[8];
    fetch_coefficients(c, vol, org, R, P.hi);

    // Evaluation
    float3 d = (float3)(0);
    float4 u2 = u1*u1;

    d.x += u2.x*(     (          (2*(c5+c7+c6+c4) + 8*(-c0))));
    d.x += u1.x*(u1.y*(          (2*(c6+c4) + 4*(-c3) + 6*(c5+c7) + 12*(-c0)))+
                      (u1.z*     (4*(c5+c7+c6+c4) + 16*(-c0))+
                            u1.w*(2*(c5+c6) + 4*(-c2) + 6*(c7+c4) + 12*(-c0))));
    d.x +=      (u2.y*(          (4*(-c0-c3+c5+c7)))+
                 u1.y*(u1.z*     (2*(c6+c4+c5+c7) + 4*(c1) + 12*(-c0))+
                            u1.w*(2*(-c3-c2+c1+c6+c4+c5) + 6*(c7) + 10*(-c0)))+
                      (u2.z*     (4*(-c0+c1))+
                       u1.z*u1.w*(2*(c7+c4+c5+c6) + 4*(c1) + 12*(-c0))+
                            u2.w*(4*(-c0-c2+c7+c4))));

    d.y += u2.x*(     (          (2*(-c6+c3+c7+c2) + 4*(-c0))));
    d.y += u1.x*(u1.y*(          (2*(-c6+c2) + 4*(-c5) + 6*(c3+c7) + 8*(-c0)))+
                      (u1.z*     (8*(-c6+c7))+
                            u1.w*(2*(-c6+c3) + 4*(-c4) + 6*(c7+c2) + 8*(-c0))));
    d.y +=      (u2.y*(          (4*(-c5-c0+c3+c7)))+
                 u1.y*(u1.z*     (2*(-c5-c0-c4-c1) + 4*(-c6) + 12*(c7))+
                            u1.w*(2*(-c5-c4-c1-c6+c3+c2) + 6*(-c0) + 10*(c7)))+
                      (u2.z*     (4*(-c6+c7))+
                       u1.z*u1.w*(2*(-c0-c1-c4-c5) + 4*(-c6) + 12*(c7))+
                            u2.w*(4*(-c0-c4+c7+c2))));

    d.z += u2.x*(     (          (2*(-c2-c4+c5+c3))));
    d.z += u1.x*(u1.y*(          (2*(-c2-c4) + 4*(-c7-c0) + 6*(c5+c3)))+
                      (u1.z*     (8*(-c4+c5))+
                            u1.w*(2*(c5+c3) + 4*(c7+c0) + 6*(-c2-c4))));
    d.z +=      (u2.y*(          (4*(-c7-c0+c5+c3)))+
                 u1.y*(u1.z*     (2*(-c7-c0-c1-c6) + 4*(-c4) + 12*(c5))+
                            u1.w*(4*(-c2-c4+c5+c3)))+
                      (u2.z*     (4*(-c4+c5))+
                       u1.z*u1.w*(2*(c7+c0+c1+c6) + 4*(c5) + 12*(-c4))+
                            u2.w*(4*(-c2-c4+c7+c0))));
    
    
    // [dx dy dz]^T_{\pi_n} = R_n*P_n*[dx dy dz]^T_{\pi_0}
    d = shuffle((float4)(d,1), P.hi).xyz;
    d *= convert_float3(R);

    return d*SCALE_G;
}
