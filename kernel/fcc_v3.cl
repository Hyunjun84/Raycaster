
#define SCALE_F (0.000086806)   // 1/11520


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
    int dR[6] = {  
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
    int3 R = (int3)((1-dR[1])*(1-dR[2])*(1-dR[5]),
                    (1-dR[0])*   dR[2] *(1-dR[4]),
                       dR[0] *   dR[1] *(1-dR[3]));
    R = dR[3]*dR[4]*dR[5] + R - R.yzx - R.zxy;
    
    // Transform p_local into the `reference left coset' (Fig 2(a)) hit_zx the reflection computed above.
    // Same as R^-1*p_local (R is one of the reflection matrices defined above)
    // Note that R^{-1}=R since R is symmetric & orthogonal.
    float3  p_ref_R = convert_float3(R)*p_local;

    // Compute the membership against the three knot planes intersecting the piece in Fig 2(a).
    // Three knot planes with their normals (-1,1,0), (-1,0,1), and (0,-1,1), respectively.
    // The input (p_ref_R) belong to one of the six tetrahedra in Fig 2(a)
    // and each piece corresponds to one of the six permutation matrices P below.
    int3 dP = (int3)(convert_int(p_ref_R.y>=p_ref_R.x),    
                     convert_int(p_ref_R.z>=p_ref_R.x), 
                     convert_int(p_ref_R.z>=p_ref_R.y));
    dP = (int3)(dP.x+dP.y, dP.x-dP.z, dP.y+dP.z);
    int3 vecPx = (int3)(dP.x==0, dP.y==1, dP.z==2);
    int3 vecPy = (int3)(dP.x==1, dP.y==0, dP.z==1);
    int3 vecPz = (int3)(dP.x==2, dP.y==-1,dP.z==0);

    
    // Compute the permutation matrix P from type_P.
    // (0,0,0):[1,0,0] (0,0,1):[1,0,0] (1,0,0):[0,1,0] (0,1,1):[0,0,1] (1,1,0):[0,1,0] (1,1,1):[0,0,1]
    //         [0,1,0]         [0,0,1]         [1,0,0]         [1,0,0]         [0,0,1]         [0,1,0]
    //         [0,0,1]         [0,1,0]         [0,0,1]         [0,1,0]         [1,0,0]         [1,0,0]
    // For p_ref_R in one of the six tetrahedral pieces, P^{-1}*p_ref_R is inside the reference tetrahedron.
    // Note that mat3 is in column-major format.

    // Transform p_ref_R into the `reference tetrahedron' hit_zx multiplying P.
    float4 p_ref = (float4)(p_ref_R, 1);

    p_ref.x = dot(p_ref_R, convert_float3(vecPx));
    p_ref.y = dot(p_ref_R, convert_float3(vecPy));
    p_ref.z = dot(p_ref_R, convert_float3(vecPz));
    
    // Fetch coefficients
    int3 dir2x = 2*R*vecPx;
    int3 dir2y = 2*R*vecPy;
    int3 dir2z = 2*R*vecPz;

    // We re-ordered the fetches such that
    // the displacement of adjacent fetches are axis-aligned.
    // Therefore, we can move to the next coefficient with low computational overhead.
    int3 offset = org;
    float c00 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 0, 0, 0)
    offset -= dir2z;
    float c15 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 0, 0,-2)
    offset += (dir2x+dir2y)>>1;
    float c19 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 1,-2)
    offset += 2*dir2z;
    float c20 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 1, 2)
    offset -= (dir2x+dir2y)>>1;
    float c16 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 0, 0, 2)    
    offset -= (dir2y+dir2z);
    float c14 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 0,-2, 0)
    offset += 2*dir2y;
    float c17 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 0, 2, 0)
    offset += (dir2x-dir2z)>>1;
    float c21 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 2,-1)
    offset += dir2z;
    float c22 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 2, 1)
    offset -= dir2y;
    float c11 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 0, 1)
    offset -= dir2x;
    float c03 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //(-1, 0, 1)
    offset -= dir2z;
    float c02 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //(-1, 0,-1)
    offset += dir2x;
    float c10 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 0,-1)
    offset += (dir2y+dir2z)>>1;
    float c12 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1, 1, 0)
    offset -= dir2x;
    float c04 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //(-1, 1, 0)
    offset -= dir2y;
    float c01 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //(-1,-1, 0)
    offset +=  dir2x;
    float c09 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 1,-1, 0)
    offset -= (dir2x+dir2z)>>1;
    float c05 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 0,-1,-1)
    offset += dir2z;
    float c06 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 0,-1, 1)
    offset += (dir2y-dir2z);
    float c07 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 0, 1,-1)
    offset += dir2z;
    float c08 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 0, 1, 1)
    offset += dir2x;
    float c26 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 2, 1, 1)
    offset -= dir2z;
    float c25 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 2, 1,-1)
    offset += (dir2z-dir2y);
    float c24 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 2,-1, 1)
    offset -= dir2z;
    float c23 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 2,-1,-1)
    offset += (dir2y+dir2z)>>1;
    float c18 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //( 2, 0, 0)
    offset -= 2*dir2x;
    float c13 = read_imagef(vol, sp2, (int4)(offset, 1)).r; //(-2, 0, 0)    

    // Evaluation
    float val = 0;
    float3 X = p_ref.xyz;
    float3 X2 = X*X;
    float3 X3 = X2*X;
    float3 X4 = X3*X;
    float3 X5 = X4*X;
    float3 X6 = X5*X;
    
    val +=    9*(X6.y*(c19+c20+c25+c26-c02-c03-c15-c16+2*(c09-c01-c18-c21-c22)+4*(c04+c14)-5*(c10+c11)+7*(c07+c08)-8*(c00+c17)+10*c12)+
                 X6.z*((c02+c03+c05+c06+c21+c22+c25+c26)-2*(c04+c09+c15+c16+c17+c18+c19+c20)+6*(c07+c08+c10+c11)-8*(c00+c12)));
    
    val +=   18*(X6.x*((c09+c10+c11+c12-c01-c02-c03-c04)+2*(c13-c18))+
                 X.x*(X5.y*((-c25-c26)+2*(c13+c15+c16-c19-c20)+3*(-c05-c06+c21+c22)+4*(-c00-c04-c10-c11-c18)+5*(c02+c03)+6*(-c01-c17)+10*(c12)+12*(c09))+
                      X5.z*((-c02+c03-c07+c08+c10-c11+c25-c26)+2*(-c05+c06+c21-c22)+3*(c15-c16-c19+c20)))+
                 X5.y*(X.z*((c19-c20)+2*(c25-c26)+3*(c10-c11+c15-c16-c21+c22)+6*(c02-c03)+7*(-c07+c08)+9*(-c05+c06))+
                       ((-c15-c16)+2*(-c10-c11-c13)+3*(c07+c08)+6*(-c02-c03+c04+c09)+9*(c05+c06)+12*(c01-c14)+16*(-c00)))+
                 X.y*X5.z*((-c05+c06+c07-c08-c10+c11+c21-c22)+2*(-c02+c03+c25-c26)+3*(c15-c16-c19+c20)));
    
    val +=   36*(X5.x*(X.y*((c14-c17)+2*(c05+c06-c07-c08)+3*(-c01+c04-c09+c12))+
                       X.z*((c15-c16)+2*(c05-c06+c07-c08)+3*(-c02+c03-c10+c11))+
                       (-c14-c15-c16-c17)+4*(-c05-c06-c07-c08)+6*(c01+c02+c03+c04+c09+c10+c11+c12-c13-c18)+16*(-c00))+
                 X.x*(6*(-c13+c18)+37*(-c01-c02-c03-c04+c09+c10+c11+c12))+
                 X.y*(6*(-c14+c17)+37*(-c01+c04-c05-c06+c07+c08-c09+c12))+
                 X.z*(6*(-c15+c16)+37*(-c02+c03-c05+c06-c07+c08-c10+c11))+
                 (c13+c14+c15+c16+c17+c18)+14*(c01+c02+c03+c04+c05+c06+c07+c08+c09+c10+c11+c12)+146*c00);
    
    val +=   45*(X2.x*(X4.y*((c05+c06+c07+c08+c15+c16+c21+c22)+2*(c01+c02+c03+c13-c14+c18+c23+c24)+3*(-c19-c20)+4*(c17-c25-c26)+5*(c10+c11)+6*(c12)+8*(-c00-c09)+10*(-c04))+
                       X4.z*((-c10-c11+c15+c16+c19+c20)+2*(c13+c14+c18+c23+c24)+3*(-c05-c06-c21-c22)+4*(-c02-c03-c04-c25-c26)+5*(c07+c08)+6*(-c09)+8*(-c00+c01)+16*(c12)))+
                 X4.y*(X2.z*((c02+c03+c21+c22)+2*(-c07-c08+c13-c14+c15+c16+c23+c24)+3*(c05+c06-c25-c26)+4*(c17-c19-c20)+6*(-c04+c10+c11)+8*(-c00+c12)+10*(-c09))+
                       (2*(c13+c15+c16+c18+c21+c22)+9*(c02+c03+c10+c11)+12*(-c04-c09+c14+c17)+13*(-c05-c06)+14*(-c01)+15*(-c07-c08)+18*(-c12)+40*(c00)))+
                 X2.y*X4.z*((-c10-c11+c15+c16+c19+c20)+2*(c13+c14+c17+c23+c24)+3*(-c02-c03-c25-c26)+4*(-c05-c06-c09-c21-c22)+5*(c07+c08)+6*(-c04)+8*(-c00+c01)+16*(c12))+
                 X4.z*(2*(c13+c14+c17+c18+c19+c20)+8*(c01+c12)+10*(c04+c09)+12*(c15+c16)+13*(-c02-c03-c05-c06)+15*(-c07-c08-c10-c11)+40*(c00)));
    
    val +=   54*X5.z*((c02-c03+c05-c06+c07-c08+c10-c11)+2*(-c15+c16));
    
    val +=   90*(X4.x*(X2.y*((c01+c04+c05+c06+c07+c08-c13+c14+c15+c16+c17+c23+c24+c25+c26)+2*(c10+c11-c19-c20-c21-c22)+3*(-c18)+5*(-c09)+7*(c12)+8*(-c00))+
                       X2.z*((c02+c03+c05+c06+c07+c08+c10+c11-c13+c14+c15+c16+c17+c23+c24+c25+c26)+2*(-c19-c20-c21-c22)+3*(-c18)+4*(-c09)+8*(-c00+c12))+
                       (c14+c15+c16+c17+c23+c24+c25+c26)+5*(c05+c06+c07+c08)+6*(c13+c18)+7*(-c01-c02-c03-c04)+9*(-c09-c10-c11-c12)+24*(c00))+
                 X2.x*(X2.y*(X2.z*((c05+c06)+2*(-c02-c03)+4*(c13+c14)+7*(c15+c16)+8*(-c18-c25-c26)+10*(c17+c23+c24)+11*(-c19-c20-c21-c22)+13*(c07+c08+c10+c11)+16*(c01)+20*(-c04)+38*(-c09)+56*(-c00)+64*(c12))+
                            ((c19+c20+c21+c22+c23+c24+c25+c26)+4*(c15+c16)+5*(-c05-c06-c07-c08-c10-c11)+6*(c12-c13-c14-c17-c18)+8*(-c02-c03)+12*(c09)+18*(c01+c04)))+
                       X2.z*((c19+c20+c21+c22+c23+c24+c25+c26)+2*(-c09)+4*(c14+c17)+5*(-c05-c06-c07-c08)+6*(-c13-c15-c16-c18)+8*(-c01-c04-c12)+9*(c10+c11)+18*(c02+c03))+
                       (-c14-c15-c16-c17)+6*(c13+c18)+8*(-c05-c06-c07-c08)+10*(c01+c02+c03+c04+c09+c10+c11+c12)+56*(-c00))+
                 X.x*(X4.y*(X.z*((-c15+c16)+2*(c05-c06-c23+c24)+3*(-c10+c11+c19-c20+c25-c26)+5*(c02-c03)+9*(-c07+c08))+
                           ((-c15-c16)+2*(c05+c06+c07+c08-c13+c18+c21+c22)+3*(c10+c11)+4*(c01)+5*(-c02-c03)+6*(-c09-c17)+8*(-c12)+10*(c04)))+
                      X.y*X4.z*(2*(-c04-c09+c13+c14-c23-c24)+3*(-c02-c03-c05-c06-c21-c22-c25-c26)+4*(c07+c08+c12+c17+c18)+8*(c01)+10*(c10+c11)+20*(-c00))+
                      X4.z*(2*(-c04-c07-c08+c09-c13-c14+c18+c19+c20)+3*(-c15-c16)+4*(c12)+6*(c05+c06)+7*(c02+c03-c10-c11)+8*(-c01)))+
                 X4.y*X.z*(2*(-c15+c16-c21+c22)+3*(-c10+c11)+5*(-c02+c03+c05-c06)+9*(c07-c08))+
                 X2.y*(X2.z*((c19+c20+c21+c22+c23+c24+c25+c26)+2*(-c04)+4*(c13+c18)+5*(-c02-c03)+6*(-c14-c15-c16-c17)+8*(-c01-c09-c10-c11-c12)+12*(c07+c08)+18*(c05+c06))+
                      ((-c13-c15-c16-c18)+6*(c14+c17)+8*(-c02-c03-c10-c11)+10*(c01+c04+c05+c06+c07+c08+c09+c12)+56*(-c00)))+
                 X.y*X4.z*(2*(c04-c09-c10-c11-c13-c14+c17+c19+c20)+3*(-c15-c16)+4*(c12)+6*(c02+c03)+7*(c05+c06-c07-c08)+8*(-c01))+
                 X2.z*((-c13-c14-c17-c18)+6*(c15+c16)+8*(-c01-c04-c09-c12)+10*(c02+c03+c05+c06+c07+c08+c10+c11)+56*(-c00)));
    
    val +=  180*(X4.x*(X.y*(X.z*((c23-c24-c25+c26)+2*(c19-c20+c21-c22)+3*(c05-c06-c07+c08)+6*(-c10+c11))+
                                (-c14+c17-c23-c24+c25+c26)+2*(c01-c04-c05-c06+c07+c08)+6*(c09-c12))+
                       X.z*((-c15+c16-c23+c24-c25+c26)+2*(c02-c03-c05+c06-c07+c08)+6*(c10-c11)))+
                 X3.x*(X3.y*((c10+c11+c15+c16-c19-c20-c21-c22-c23-c24-c25-c26)+2*(c01-c04+c07+c08+c12+c17)+4*(c18)+8*(-c00))+
                       X2.y*(X.z*((-c15+c16)+3*(c19-c20+c21-c22-c23+c24+c25-c26)+4*(c02-c03)+5*(-c10+c11)+6*(c05-c06)+12*(-c07+c08))+
                                 (-c10-c11+c19+c20+c21+c22)+2*(-c09+c13-c15-c16+c23+c24+c25+c26)+4*(-c01-c04)+6*(-c18)+8*(-c12)+16*(c00))+
                       X.y*(X2.z*(2*(-c09+c14)+3*(c10+c11+c15+c16-c19-c20-c21-c22-c23-c24-c25-c26)+4*(c01-c04+c17)+6*(c07+c08)+8*(c12)+12*(c18)+24*(-c00))+
                                 (-c05-c06+c07+c08-c23-c24+c25+c26)+4*(c09-c12))+
                       X3.z*((-c10+c11-c15+c16+c19-c20+c21-c22-c23+c24+c25-c26)+2*(c02-c03+c05-c06)+4*(-c07+c08))+
                       X2.z*((c19+c20+c21+c22)+2*(c09+c13-c14-c17+c23+c24+c25+c26)+4*(-c02-c03-c12)+5*(-c10-c11)+6*(-c18)+16*(c00))+
                       X.z*((-c05+c06-c07+c08-c23+c24-c25+c26)+4*(c10-c11)))+
                 X2.x*(X3.y*(X.z*(2*(c21-c22)+3*(c05-c06+c23-c24+c25-c26)+4*(c19-c20)+6*(c02-c03)+9*(-c07+c08)+16*(-c10+c11))+
                            ((-c05-c06-c07-c08-c15-c16+c19+c20-c23-c24+c25+c26)+2*(c09+c14+c21+c22)+4*(-c01+c04-c17)+8*(c00-c12)))+
                       X2.y*X.z*((-c15+c16)+4*(-c19+c20)+5*(-c05+c06-c21+c22-c23+c24-c25+c26)+8*(-c02+c03)+13*(c07-c08)+25*(c10-c11))+
                       X.y*(X3.z*(2*(-c15+c16+c19-c20)+3*(c23-c24+c25-c26)+4*(c21-c22)+5*(c05-c06)+6*(c02-c03)+7*(-c07+c08)+14*(-c10+c11))+
                            X2.z*((c05+c06)+2*(-c17)+3*(-c15-c16)+4*(-c14+c21+c22)+5*(c19+c20-c23-c24+c25+c26)+6*(-c10-c11)+7*(-c07-c08)+8*(-c01+c04)+22*(c09)+24*(c00)+28*(-c12))+
                            X.z*((-c19+c20-c21+c22)+2*(c23-c24-c25+c26)+3*(c10-c11)+4*(-c05+c06+c07-c08)))+
                       X3.z*((-c21+c22-c23+c24-c25+c26)+2*(-c19+c20)+3*(-c05+c06+c07-c08+c15-c16)+4*(-c02+c03)+5*(c10-c11)))+
                 X.x*(X3.y*(X2.z*((c02+c03-c05-c06)+2*(c13-c21-c22)+3*(c15+c16-c19-c20-c23-c24)+4*(c01+c07+c08+c09+c17-c25-c26)+5*(c10+c11)+6*(-c04)+8*(c18)+12*(c12)+28*(-c00))+
                            ((-c05-c06-c07-c08+c10+c11+c21+c22)+4*(c00-c12)))+
                      X2.y*(X3.z*(2*(-c15+c16+c19-c20)+3*(c21-c22-c23+c24)+4*(-c10+c11+c25-c26)+5*(c02-c03)+6*(c05-c06)+11*(-c07+c08))+
                            X2.z*((c02+c03)+3*(-c15-c16)+4*(-c09-c13+c23+c24+c25+c26)+5*(c19+c20+c21+c22)+6*(-c17)+8*(-c01-c18)+10*(c04-c10-c11)+12*(-c07-c08)+16*(-c12)+48*(c00))+
                            X.z*((-c19+c20-c23+c24-c25+c26)+2*(-c21+c22)+4*(-c02+c03+c10-c11)+6*(c07-c08)))+
                      X.y*X2.z*((c21+c22-c23-c24+c25+c26)+2*(c19+c20)+3*(-c10-c11)+4*(-c01+c04+c09-c12)+6*(-c07-c08)+12*(c00))+
                      X3.z*((-c05+c06+c07-c08-c19+c20)+2*(c10-c11)))+
                 X3.y*(X3.z*((-c15+c16+c19-c20+c21-c22+c23-c24+c25-c26)+2*(c02-c03+c05-c06-c07+c08)+5*(-c10+c11))+
                       X2.z*((c19+c20-c23-c24+c25+c26)+2*(-c07-c08-c13+c14+c21+c22)+4*(-c05-c06+c09-c17)+6*(c04)+8*(c00-c12))+
                       X.z*((-c02+c03-c21+c22)+2*(c07-c08)))+
                 X3.z*(X2.y*((-c21+c22-c23+c24-c25+c26)+2*(c07-c08-c19+c20)+3*(-c02+c03+c15-c16)+4*(-c05+c06)+6*(c10-c11))+
                        X.y*((-c02+c03+c10-c11-c19+c20)+2*(c07-c08))));

    val +=  360*(X3.x*(X.y*X.z*((-c19+c20-c21+c22)+2*(c23-c24-c25+c26)+3*(c10-c11)+4*(-c05+c06+c07-c08))+
                      (c01+c02+c03+c04-c09-c10-c11-c12)+2*(-c13+c18))+
                 X2.x*(X.y*((c14-c17)+4*(c05+c06-c07-c08)+5*(-c01+c04-c09+c12))+
                       X.z*((c15-c16)+4*(c05-c06+c07-c08)+5*(-c02+c03-c10+c11)))+
                 X.x*(X3.y*X.z*((-c05+c06+c10-c11-c19+c20+c23-c24-c25+c26)+2*(-c21+c22)+3*(-c02+c03)+7*(c07-c08))+
                      X2.y*((c13-c18)+4*(c02+c03-c10-c11)+5*(-c01-c04+c09+c12))+
                      X.y*X3.z*((-c21+c22+c23-c24-c25+c26)+2*(c10-c11+c15-c16-c19+c20)+3*(-c02+c03-c05+c06)+5*(c07-c08))+
                      X2.z*((c13-c18)+4*(c01+c04-c09-c12)+5*(-c02-c03+c10+c11)))+
                 X3.y*((c01-c04+c05+c06-c07-c08+c09-c12)+2*(-c14+c17))+
                 X2.y*X.z*((c15-c16)+4*(c02-c03+c10-c11)+5*(-c05+c06-c07+c08))+
                 X.y*X2.z*((c14-c17)+4*(c01-c04+c09-c12)+5*(-c05-c06+c07+c08))+
                 X3.z*((c02-c03+c05-c06+c07-c08+c10-c11)+2*(-c15+c16)));
    
    val += 2880*(X.x*X.y*(c01-c04-c09+c12)+
                 X.x*X.z*(c02-c03-c10+c11)+
                 X.y*X.z*(c05-c06-c07+c08));
    
    return val*SCALE_F;
}

