//*********
//风格化金属Shader
//*********
//*********
//*********

Shader "PaxtonLiu/MetallicPBRStyle"
{
    Properties
    {
        [MainTexture]_MainTex("Color" , 2D) = "white"{}
        [Maincolor]_ColorTint("Tint" , color) = (1,1,1,1)
        _AlbedoInt("_AlbedoInt" , Range( 0 , 4)) = 1
        [Header(RM)]
        _RMMap("R:Routhness G:Matelness B:LighMap A:",2D) = "white"{}
        _Routhness("Routhness", Range( 0 , 1)) = 1
        _roughnessADD("RoughnessADD", Range( -1 , 1)) = 0
        _Matelness("Matelness", Range( 0 , 1)) = 1
        _MetalIntensity("_MetalIntensity", Range( 0 , 1)) = 0
        _RampTex("RampTex",2D) = "white"{}
        _RampTexSpec("RampTexSpec",2D) = "white"{}
        _MetalMap("MetalFactor",2D) = "Black"{}
        _RimMap("RimMap",2D) = "Black"{}
        _RimIntensity("_RimIntensity"  , Range( 0 , 2)) = 1
        _NormalMap("Normal贴图"  ,2D) = "bump"{}
        _NormalIntensity("NormalIntensity"  , Range( 0 , 2)) = 1
        _EnvSpecInt("EnvSpecInt", Range( 0 , 10)) = 1
        _Cubemap("_Cubemap"  ,CUBE) = "black"{}
        _lightMap("lightMap"  ,2D) = "black"{}
        _specInt("_specInt"  ,Range( 0 , 10)) = 1
        _specHardness("_specHardness"  ,Range( 0 , 0.9)) = 0.8
        _env_specInt("_env_specInt"  ,Range( 0 , 10)) = 1
        _HardSpecInt1("_HardSpecInt1"  ,Range( 0 , 10)) = 1
        _HardSpecInt2("_HardSpecInt2"  ,Range( 0 , 10)) = 1
        _rampCol2("rampCol2" , 2D) = "white"{}  
         [Header(GBVS)]
        _BaseMap("BaseMap" , 2D) = "white"{}
        _SSSMap      ("SSMap" , 2D) = "black"{}
        _ILMMap      ("ILMMap" , 2D) = "white"{}  
        _ShadowThreshold     ("ShadowThreshold"  ,Range( 0 , 1)) = 1
        _ShadowHardness      ("ShadowHardness"  ,Range( 0 , 10)) = 1
        _SpecSize("_SpecSize"  ,Range( 0 , 1)) = 1
        _SpecSize2("_SpecSize2"  ,Range( 0 , 1)) = 1
        _SpecColor("_SpecColor" , color) = (1,1,1,1)
        _SpecDarkColor("_SpecDarkColor" , color) = (1,1,1,1)
        _SpecIntensity("_SpecIntensity"  ,Range( 0 , 10)) = 1
        _RimLightDir("_RimLightDir"  ,Float) = (1,1,1,0)
        _RimLightColor("_RimLightColor" , color) = (1,1,1,1)
        _SmoothstepMin("_SmoothstepMin"  ,Range( 0 , 10)) = 1
        _SmoothstepMax("_SmoothstepMax"  ,Range( 0 , 10)) = 1

    }
    
    SubShader
    {
        Tags{"RenderPipeline" = "UniversalPipeline"}

        Pass
        {
            Name "ForwardLit" // For debugging
            Tags{"LightMode" = "UniversalForward"} // Pass specific tags. 
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"
            

            struct appdata
            {
                half3 vertex : POSITION;
                half2 uv : TEXCOORD0;
                half4 normalOS : NORMAL;
                half4 tangent : TANGENT;
                half4 color : COLOR;
            };

            struct v2f
            {
                half4 posCS : SV_POSITION;
                half2 uv : TEXCOORD0;
                half3 posWS : TEXCOORD1;
                half3 normalWS : TEXCOORD3;
				half3 tangentWS: TEXCOORD4;
				half3 binormalWS: TEXCOORD5;
                half3 normalVS_uv: TEXCOORD6;
                half4 vertex_color : TEXCOORD7;

            };

            CBUFFER_START(UnityPerMaterial)
            half _AlbedoInt;
            half _specularPowIint;
            half _RampRange;
			half _MetalIntensity;
			half _AlbedoIntensity;
            half _Bump;
            half _Routhness;
            half _roughnessADD;
            half _Matelness;
            half _NormalIntensity;
            half _EnvSpecInt;
            half _specInt;
            half _specHardness;
            half _env_specInt;
            half _HardSpecInt1;
            half _HardSpecInt2;
            half _ShadowThreshold;
            half _ShadowHardness;
            half _SpecSize;
            half3 _SpecColor;
            half3 _SpecDarkColor;
            half _SpecIntensity;
            half3 _RimLightDir;
            half3 _RimLightColor;
            half _SmoothstepMin;
            half _SmoothstepMax;
            half _RimIntensity;

            
			CBUFFER_END

            sampler2D _MainTex; half4 _MainTex_ST;
            half4 _ColorTint;
            samplerCUBE _Cubemap;
            sampler2D _RMMap; 
            sampler2D _MetalMap;
            sampler2D _RimMap;
            sampler2D _NormalMap;
            sampler2D _RampTex;
            sampler2D _RampTexSpec;
            sampler2D _lightMap; half4 _lightMap_ST;
            sampler2D _rampCol2; 
            sampler2D _BaseMap; 
            sampler2D _SSSMap; 
            sampler2D _ILMMap; 
             
            
            
            
            
            
           

             half3 NPR_Base_Ramp (half NdotL,sampler2D RampTex, half4 lightMap, half RampRange)
            {
                half halfLambert = smoothstep(0,0.5,NdotL) * lightMap.g * 2;
                //halfLambert = smoothstep(0,0.5,NdotL) * 1;
                 //halfLambert = NdotL * 0.5 + 0.5;
                 //halfLambert = NdotL;

                /* 
                Skin = 1.0
                Silk = 0.7
                Metal = 0.5
                Soft = 0.3
                Hand = 0.0
                */
                
                    return tex2D(RampTex ,half2(halfLambert, RampRange)).rgb;//因为分层材质贴图是一个从0-1的一张图 所以可以直接把他当作采样UV的Y轴来使用 
                    return halfLambert; 
                    return NdotL; 
                    //又因为白天需要只采样Ramp贴图的上半段，所以把他 * 0.45 + 0.55来限定范围 (范围 0.55 - 1.0)
            }
            
            half3 NPR_Base_Specular(half NdotL,half NdotH ,sampler2D RampTex,half3 baseColor,half4 lightMap,half3 MetalFactor)
            {
                half Ks = 0.04;
                half  SpecularPow = exp2(0.5 * lightMap.r * 11.0 + 2.0);//这里乘以0.5是为了扩大高光范围
                half  SpecularNorm = (SpecularPow+8.0) / 8.0;
                half3 SpecularColor =  baseColor * lightMap.b;
                 SpecularColor = lerp(half3(0.3, 0.3, 0.3), baseColor,  0.1) * lightMap.b;
                half SpecularContrib = baseColor * (SpecularNorm * pow(NdotH, SpecularPow));
                 
                 half halfLambert = max(0,NdotL * 0.5 + 0.5);
                half SpecRamp = tex2D(RampTex ,half2(halfLambert, 0.2)).rgb;
                 
                 MetalFactor *= lerp(_MetalIntensity * 5,_MetalIntensity  * 10, lightMap.r) ;
                half3 MetalColor = MetalFactor * baseColor * step(0.95,lightMap.r);
                return MetalColor * SpecularColor ;
                return MetalFactor;

            }


            half RoughnessToSpecularExponent(half roughness)
            {
               return  sqrt(2 / (roughness + 2));
            }
            
            v2f vert (appdata v)
            {
                v2f o ;
                //posOS=>CS  "ShaderVariablesFunctions.hlsl"
                VertexPositionInputs posnInputs = GetVertexPositionInputs(v.vertex);
                o.posCS = posnInputs.positionCS;
                o.posWS = posnInputs.positionWS;
                //o.posCS = TransformObjectToHClip(v.vertex);
                
                //法线OS=>WS "ShaderVariablesFunctions.hlsl"
                VertexNormalInputs normalInputs = GetVertexNormalInputs(v.normalOS.xyz);
                o.normalWS = normalize(normalInputs.normalWS);
                 half3 normalVS = mul(UNITY_MATRIX_IT_MV, v.normalOS).xyz;
                 o.normalVS_uv.xy = normalize(normalVS).xy * 0.5 + 0.5;
                o.tangentWS = normalize(mul(UNITY_MATRIX_M, half4(v.tangent.xyz, 0.0)).xyz);
                 o.binormalWS = normalize(cross(o.normalWS, o.tangentWS) * v.tangent.w); // tangent.w is specific to Unity
                o.uv = TRANSFORM_TEX(v.uv ,_MainTex);
                 o.vertex_color = v.color;

                return o;
            }
            //D
                float Distribution(float roughness , float ndoth)
                {
	                float lerpSquareRoughness = pow(lerp(0.002, 1, roughness), 2);
	                float D = lerpSquareRoughness / (pow((pow(ndoth, 2) * (lerpSquareRoughness - 1) + 1), 2) * PI);
	                return D;
                }
            //G
                float Geometry(float roughness , float ndotl , float ndotv)
                {
                    float k = pow(roughness + 1, 2) / 8;
                    k = max(k,0.5);
                    float GLeft = ndotl / lerp(ndotl, 1, k);
                    float GRight = ndotv / lerp(ndotv, 1, k);
                    float G = GLeft * GRight;
                    return G;
                }
            //F
                float3 FresnelEquation(float3 F0 , float ldoth)
                {
                    float3 F = F0 + (1 - F0) * pow((1.0 - ldoth),5);
                    return F;
                }
                
            half4 frag (v2f i) : SV_TARGET
            {
                
                half4 shadowCoord = TransformWorldToShadowCoord(i.posWS);
                Light light = GetMainLight(shadowCoord);
                half3 vDirWS = normalize(_WorldSpaceCameraPos.xyz - i.posWS.xyz);
                half3 reflectDir = reflect(-vDirWS, i.normalWS);
                half3 nDirTS = UnpackNormal(tex2D(_NormalMap,i.uv));
                nDirTS = lerp(float3(0,0,1),nDirTS,_NormalIntensity);
                half3 nDirVS = TransformWorldToView(vDirWS);
                
                half3x3 tangent2World = half3x3(i.tangentWS,i.binormalWS ,i.normalWS);
                half3 nDirWS = normalize(mul(nDirTS,tangent2World));
                
                half3 matCapUV = reflectDir * 0.5 + 0.5;
                half3 ViewN =  i.normalVS_uv;
                ViewN.xy += nDirTS.xy * _Bump;
                
                half NdotL = dot(nDirWS,normalize(light.direction));
                NdotL = max(0,NdotL);
                half NdotV = dot(nDirWS,vDirWS);
                half NdotH = dot(nDirWS,normalize(light.direction + vDirWS));
                NdotH = max(0,NdotH);
                half LdotH = dot(normalize(light.direction),normalize(light.direction + vDirWS));
                
                half halfLambert = smoothstep(0,0.5,NdotL);
                
                // sample the texture
                half4 Albedo  = tex2D(_MainTex, i.uv).rgba * _ColorTint ;
                half4 RMCol  = tex2D(_RMMap, i.uv);
                half roughness = (1 - RMCol.a ) ;
                roughness = lerp(roughness,1,_Routhness);
                roughness += _roughnessADD;
                roughness = max(0.05,roughness);
                half Matelness = RMCol.r ;
                //lightMap.b = 1;
                
                //硬高光主光镜面反射
                // half Var_SpecRamp = tex2D(_RampTexSpec ,pow(NdotV,5) );
                // half HardSpec = lightMap.b * Var_SpecRamp * step(0.95,lightMap.r) * _HardSpecInt1 ;
                // HardSpec += smoothstep(0.1,0.2,lightMap.b * pow(NdotVOffset,100) * step(0.95,lightMap.r) )* _HardSpecInt2 ;
                // //HardSpec = smoothstep(0.1,0.2,HardSpec) * _HardSpecInt ;
  
                //主光漫反射
                //下式是根据物理能量守恒计算的kd系数，F是镜面反射公式的F，但效果不如内置宏好
                //float3 kd = (1 - F)*(1 - metallic);  
                float3 kd = OneMinusReflectivityMetallic(RMCol.g);
                //atten是阴影衰减系数，平行光不衰减都是1，点光射光衰减
                float atten = light.distanceAttenuation ;
                //_LightColor0是该pass内置光源颜色
                float3 diffColor = kd * Albedo * light.color * NdotL * atten;

                
                //这里默认一个环境光基数为0.04，unity内置了一个预定义unity_ColorSpaceDielectricSpec，但与0.04有点差距，因此我们使用0.04
                //half3 F0 = unity_ColorSpaceDielectricSpec.rgb;
                half3 F0 = half3(0.04,0.04,0.04);
                F0 = lerp(F0, Albedo, Matelness);
                float3 F = FresnelEquation(F0 , LdotH);
                float D = Distribution(roughness , NdotH);
                float G = Geometry(roughness , NdotL , NdotV);

                float3 Specular = F*D*G/(4 * NdotV * NdotL + 0.001);//一定要有这个+0.001，来防止边缘处除以0导致过曝
                float3 specColor = Specular * PI * light.color * NdotL * atten;
                //直接光照结果
                float3 DirectLightResult = diffColor + specColor;


                //间接光照漫反射
                float3 ambient = 0.03 * Albedo;//基础的环境光
                float3 irradiance = SampleSH(float4(nDirWS,1));//内置球谐光照计算相应的采样数据
                float3 iblDiffuse = max(float3(0,0,0),irradiance + ambient.rgb);
                float3 Flast = F0 + (max(float3(1 ,1, 1)*(1 - roughness), F0) - F0) * pow(1.0 - NdotV, 5.0);
                float3 iblKd = (float3(1.0,1.0,1.0) - Flast) * (1 - Matelness);//间接光漫反射系数
                iblDiffuse *= iblKd * Albedo;

                //间接镜面反射
                // float mip_roughness = perceptualRoughness * (1.7 - 0.7 * perceptualRoughness);
                // float3 reflectVec = reflect(-viewDir, i.normal);
                //
                // half mip = mip_roughness * UNITY_SPECCUBE_LOD_STEPS;
                // half4 rgbm = UNITY_SAMPLE_TEXCUBE_LOD(unity_SpecCube0, reflectVec, mip); 
                //
                // float3 iblSpecular = DecodeHDR(rgbm, unity_SpecCube0_HDR);
                //
                // float3 IndirectResult = iblDiffuse + iblSpecular;
                half3 var_Cubemap = texCUBElod(_Cubemap, float4(reflectDir, lerp(8.0, 0.0, roughness))).rgb;
                half reflectInt = Matelness * 1;
                half3 specCol = F;
                half3 envSpec = 1 * reflectInt * var_Cubemap * _EnvSpecInt;
                
                half pbrGrayCol = DirectLightResult + 0 + envSpec ;
                
                half3 rampCol = tex2D(_RampTex,float2(pbrGrayCol,0.2)) ;
                
                //高光点
                half4 lightMap = tex2D(_lightMap,i.uv);
                half Var_SpecRamp = tex2D(_RampTexSpec ,pow(NdotV,5) );
                half HardSpec = roughness * Var_SpecRamp * step(0.95,lightMap.r) * _HardSpecInt1 ;
                half HardSpec2_term = smoothstep(0.1,1-_specHardness,lightMap.b * pow(NdotV,500* (1 - _SpecSize)) * step(0.95,lightMap.r) ) ;
                HardSpec2_term = saturate(HardSpec2_term * 10 );
                HardSpec += HardSpec2_term * _HardSpecInt2 ;
                HardSpec *= NdotL ;
                //HardSpec = smoothstep(0.1,0.2,HardSpec) * _HardSpecInt ;

                
                //原神高光
                //half3 rampCol2 = tex2D(_rampCol2,i.uv);
                half metalFact = tex2D(_MetalMap,matCapUV).r;
                half3 SpecYS = NPR_Base_Specular(NdotL,NdotH,_rampCol2,Albedo,lightMap,metalFact);
                pbrGrayCol = pbrGrayCol * _env_specInt;
                SpecYS = SpecYS + _env_specInt;

                //MatCap菲涅尔
                half RimColor = tex2D(_RimMap,matCapUV).r * _RimIntensity;
                
                
                //***********************************
                //GBVS
                half4 var_base = tex2D(_BaseMap,i.uv);
                half4 var_sss = tex2D(_SSSMap,i.uv);
                half4 var_ilm = tex2D(_ILMMap,i.uv);
                
                half spec_intensity = var_ilm.r;
                half shadow_control = var_ilm.g * 2 - 1 ;//0~0.5 => -1 到 0
                half spec_size = var_ilm.b;
                half inner_line = var_ilm.a;
                half ao = i.vertex_color.r;
                //diff
                half lambert_term = halfLambert * ao ; //shadow_control就是上面所说的阴影倾向权重
                half toon_diffuse = saturate((lambert_term - _ShadowThreshold) * _ShadowHardness);  //色阶化处理得到亮暗部分mask
                half3 GBVSdiff = lerp(var_sss, var_base, toon_diffuse);//进行混合

                    //***************NPR_Base_Specular_Test************
                    half3 NPRSpecTest = lerp(SpecYS * 0.5 + _SpecDarkColor * 0.5, SpecYS, toon_diffuse);//进行混合
                    //NPRSpecTest = lerp(SpecYS * 0.2, SpecYS, toon_diffuse);//进行混合
                
                //spec
                half spec_term = (NdotV + 1.0) * 0.5 * ao + shadow_control;
                spec_term = halfLambert * 0.9 + spec_term * 0.1;
                //spec_term = spec_term * 0.1 + halfLambert * 0.9;   //反射所占权重
                half toon_spec = saturate((spec_term - (1 - spec_size * _SpecSize)) * 500);//色阶化处理得到高光mask
                half3 spec_col = _SpecColor.xyz * 0.6 + var_base * 0.4; //希望高光颜色带有basecolor的倾向
                half3 GBVSspec = toon_spec * spec_col * spec_intensity * _SpecIntensity;//进行混合
                //rimcol
                float3 rimlight_dir = normalize(mul(UNITY_MATRIX_V, _RimLightDir.xyz));//转换到相机空间
                half rim_lambert = (dot(nDirVS, rimlight_dir) + 1.0) * 0.5;//从-1.0-1.0映射到0.0-1.0                 
                half rimlight_term = halfLambert * ao + shadow_control;//边缘光因子
                half toon_rim = saturate((rim_lambert - _ShadowThreshold) * 20);
                half3 rim_color = (_RimLightColor + var_base) * 0.5 * var_sss;//sss_mask区分边缘光区域的强度
                half3 GBVSrim = toon_rim * rim_color  * toon_diffuse;//base_mask区分皮肤与非皮肤区域，看自己喜欢选择乘不乘
                //rimdark
                half rimdark_term = smoothstep(_SmoothstepMin,_SmoothstepMax,NdotV);
                rimdark_term = lerp( 1 ,rimdark_term, 1- toon_diffuse);
                half3 GBVSrim2 = rimdark_term * _RimLightColor;
                half4 Test = 1;
                Test.rgb =   rampCol + pbrGrayCol;
                Test.rgb =  SpecYS *_AlbedoInt+HardSpec ;
                Test.rgb =  NPRSpecTest *_AlbedoInt + HardSpec + RimColor +
                    Albedo * ( 1 - RMCol.r ) *_AlbedoInt;
                //Test.rgb =  SpecYS + RimColor;
                //Test.rgb =  NPRSpecTest;
                
                half4 finalcol = 1;
                finalcol.rgb = rampCol * Albedo* _AlbedoInt+HardSpec;
                
                half4 metallicPbrStyle =1;
                metallicPbrStyle.rgb = pbrGrayCol * Albedo;

                half4 GBVScolpr = 1;
                GBVScolpr.rgb = GBVSspec + GBVSdiff;
                GBVScolpr.rgb = GBVSrim2;
                GBVScolpr.rgb = lerp(_RimLightColor * 0.4 + var_sss * 0.6, GBVSspec + GBVSdiff , rimdark_term);
                //GBVScolpr.rgb = GBVSspec;
                // apply fog
                //UNITY_APPLY_FOG(i.fogCoord, finalcol);
                return Test;
                return GBVScolpr ;
                return metallicPbrStyle;
                return finalcol;
            }
            ENDHLSL
        }
    }
}
