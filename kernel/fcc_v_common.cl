
__inline int sgn(float x) 
{
    return convert_int(x>=0.0f)*2-1;
}

__inline int3 find_origin(float3 p)
{
    int3 org = convert_int3(round(p));
    if( (convert_int(org.x+org.y+org.z)&0x01) != 0 ) {
        float3 l = p-convert_float3(org);
        float3 ul = fabs(l);
        if(ul.x > ul.y) {
            if(ul.x > ul.z)   org.x += sgn(l.x);
            else              org.z += sgn(l.z);
        } else {
            if(ul.y > ul.z)   org.y += sgn(l.y);
            else              org.z += sgn(l.z);
        }
    }
    return org;
}

__inline int3 find_R(float3 l)
{
    // computes the membership against the six knot planes intersecting the unit cube centered at the local origin
    int dR[6] = {  
        convert_int(l.z>=l.y),
        convert_int(l.z>=l.x),
        convert_int(l.y>=l.x),
        convert_int(l.x>=-l.y),
        convert_int(l.x>=-l.z),
        convert_int(l.y>=-l.z)
    };

    // type_R: the `reflection transformation' which is one of four `even reflections'
    // The reflection matrix R for each type:
    // (0,0,0): 1, 0, 0] (0,1,1): 1, 0, 0] (1,0,1): [-1, 0, 0] (1,1,0): [-1, 0, 0]
    //          0, 1, 0]          0,-1, 0]          0, 1, 0]          0,-1, 0]
    //          0, 0, 1]          0, 0,-1]          0, 0,-1]          0, 0, 1]
    int3 R = (int3)((1-dR[1])*(1-dR[2])*(1-dR[5]),
                    (1-dR[0])*   dR[2] *(1-dR[4]),
                       dR[0] *   dR[1] *(1-dR[3]));
    R += dR[3]*dR[4]*dR[5] - R.yzx - R.zxy;
    return R;
}

__inline uint8 find_P(float3 l)
{
    int3 dP = (int3)(convert_int(l.y>=l.x),    
                     convert_int(l.z>=l.x), 
                     convert_int(l.z>=l.y));

    int idx_P = 2*(dP.x+dP.y)+dP.z;
    uint3 vecP1 = (uint3)(convert_uint(idx_P==0), convert_uint(idx_P==4), convert_uint(idx_P==3));
    uint3 vecP2 = (uint3)(convert_uint(idx_P==1), convert_uint(idx_P==2), convert_uint(idx_P==5));

    return (uint8)(vecP1.yxz+vecP2.yzx + 2*(vecP1.zyx+vecP2.zxy), 3,
                   vecP1.zxy+vecP2.yzx + 2*(vecP1.yzx+vecP2.zxy), 3);
}

__inline float4 to_barycentric(float3 p)
{
    return (float4)(1.0f-p.x-p.y, 
                         p.y+p.z, 
                         p.x-p.y, 
                         p.y-p.z);
}