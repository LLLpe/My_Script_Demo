Shader "PaxtonLiu/Paxton_Cartoon"
{
    Properties
    {
    	[KeywordEnum(Body,Face)] _ShaderEnum("Shader类型", int) = 0
        [MainTexture]_MainTex("Color" , 2D) = "white"{}
        _lightMap("lightMap"  ,2D) = "black"{}
    	_RampTexSpec("RampTexSpec",2D) = "white"{}
//        [Maincolor]_ColorTint("Tint" , color) = (1,1,1,1)
//        _AlbedoInt("_AlbedoInt" , Range( 0 , 4)) = 1
        _Color("_Color", COLOR) = (1.00, 1.00, 1.00, 1.00)
		_Color2("_Color2", COLOR) = (1, 1, 1, 1)
		_Color3("_Color3", COLOR) = (1, 1, 1, 1)
		_Color4("_Color4", COLOR) = (1, 1, 1, 1)
		_Color5("_Color5", COLOR) = (1, 1, 1, 1)
        
	    [Header(Shadow)]
//        _ShadowRampToggle("_ShadowRampToggle", Float) = 0
        _ShadowThreshold     ("阴影范围"  ,Range( 0.2 , 0.8)) = 0.75
        _ShadowHardness      ("阴影硬度"  ,Range( 5, 15)) = 10
//		_ShadowRampWidth("_ShadowRampWidth", Float) = 1
//		_LightArea("_LightArea", Range(0, 1)) = 0.5
		_ShadowMultColor("_ShadowMultColor", COLOR) = (0.94731, 0.52712, 0.45641, 1.0)
		_ShadowMultColor2("_ShadowMultColor2", COLOR) = (0.53405, 0.35817, 0.34012, 1.0)
		_ShadowMultColor3("_ShadowMultColor3", COLOR) = (0.7816, 0.7816, 0.7816, 1.0)
		_ShadowMultColor4("_ShadowMultColor4", COLOR) = (0.78741, 0.44799, 0.52252, 1.0)
		_ShadowMultColor5("_ShadowMultColor5", COLOR) = (0.78741, 0.44799, 0.52252, 1.0)
    	
        
        _MetalIntensity("金属度强度", Range( 0 , 1)) = 0
        _MetalMap("MetalFactor",2D) = "Black"{}
        _NormalMap("Normal贴图"  ,2D) = "bump"{}
    	_NormalIntensity("法线强度",Range(0,1)) = 1

        _HardSpecInt("高光1"  ,Range( 0 , 5)) = 0.8

        _HardSpecInt2("高光2"  ,Range( 0 , 5)) = 0.8
        
         
//        _SpecDarkColor("_SpecDarkColor" , color) = (1,1,1,1)
//        _SpecIntensity("_SpecIntensity"  ,Range( 0 , 10)) = 1
//        _RimLightDir("_RimLightDir"  ,Float) = (1,1,1,0)
//        _RimLightColor("_RimLightColor" , color) = (1,1,1,1)
//        _SmoothstepMin("_SmoothstepMin"  ,Range( 0 , 10)) = 1
//        _SmoothstepMax("_SmoothstepMax"  ,Range( 0 , 10)) = 1

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

            half4 _Color;
			half4 _Color2;
			half4 _Color3;
			half4 _Color4;
			half4 _Color5;

            float3 _ShadowMultColor;
			float3 _ShadowMultColor2;
			float3 _ShadowMultColor3;
			float3 _ShadowMultColor4;
			float3 _ShadowMultColor5;
            
			half _MetalIntensity;
            half _Bump;
            half _Routhness;

            half _Matelness;
            half _NormalIntensity;
            half _env_specInt;
            half _ShadowThreshold;
            half _ShadowHardness;
            
            half _HardSpecInt;
            // half _SpecSize;
            // half _specHardness;
            half _HardSpecInt2;
            // half _SpecSize2;
            // half _specHardness2;
            
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
            sampler2D _MetalMap;
            // sampler2D _RimMap;
            sampler2D _NormalMap;
            sampler2D _RampTex;
            sampler2D _RampTexSpec;
            sampler2D _lightMap; half4 _lightMap_ST;
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
            
            half3 NPR_Base_Specular(half NdotL,half NdotH ,sampler2D RampTex,half3 baseColor,half metallic, half specular,half3 MetalFactor)
            {
                half Ks = 0.04;
                half  SpecularPow = exp2(0.5 * metallic * 11.0 + 2.0);//这里乘以0.5是为了扩大高光范围
                half  SpecularNorm = (SpecularPow+8.0) / 8.0;
                half3 SpecularColor =  baseColor * specular;
                 SpecularColor = lerp(half3(0.3, 0.3, 0.3), baseColor,  0.1) * specular;
                half SpecularContrib = baseColor * (SpecularNorm * pow(NdotH, SpecularPow));
                 
                 half halfLambert = max(0,NdotL * 0.5 + 0.5);
                half SpecRamp = tex2D(RampTex ,half2(halfLambert, 0.2)).rgb;
                 
                 MetalFactor *= lerp(_MetalIntensity * 5,_MetalIntensity  * 10, metallic) * specular * 0.3 ;
                half3 MetalSpecColor = MetalFactor * baseColor * step(0.95,metallic);
                return MetalSpecColor * 1 ;
                return MetalFactor;

            }

            half3 NPR_Base_Specular2(half NdotL,half NdotH ,sampler2D RampTex,half3 baseColor,half metallic, half specular,half3 MetalFactor)
            {
                half Ks = 0.04;
                half  SpecularPow = exp2(0.5 * metallic * 11.0 + 2.0);//这里乘以0.5是为了扩大高光范围
                half  SpecularNorm = (SpecularPow+8.0) / 8.0;
                half3 SpecularColor =  baseColor * specular;
                 SpecularColor = lerp(half3(0.3, 0.3, 0.3), baseColor,  0) *specular;
                half SpecularContrib = baseColor * (SpecularNorm * pow(NdotH, SpecularPow));
                 
                 half halfLambert = max(0,NdotL * 0.5 + 0.5);
                half SpecRamp = tex2D(RampTex ,half2(halfLambert, 0.2)).rgb;
                //原神的金属贴图（这里我使用了一种拟合曲线来模拟）
                // half MetalDir = normalize(mul(UNITY_MATRIX_V,normalDir));
                // half MetalRadius = saturate(1 - MetalDir) * saturate(1 + MetalDir);
                // half MetalFactor = saturate(step(0.5,MetalRadius)+0.25) * 0.5 * saturate(step(0.15,MetalRadius) + 0.25)  ;
                //简单方法是直接采样
                 MetalFactor *= lerp(_MetalIntensity * 5,_MetalIntensity  * 10, metallic) ;
                half3 MetalColor = MetalFactor * baseColor * step(0.95,metallic);
                return MetalColor * SpecularColor ;
                return SpecularColor.b ;
                return MetalColor;
                //return SpecularColor * (SpecularContrib  * NdotL* Ks * lightMap.g + MetalColor) ;
                return SpecRamp;
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
                
            half4 frag (v2f i) : SV_TARGET
            {
                
                half4 shadowCoord = TransformWorldToShadowCoord(i.posWS);
                Light light = GetMainLight(shadowCoord);
                half3 vDirWS = normalize(_WorldSpaceCameraPos.xyz - i.posWS.xyz);
                half3 nDirTS = UnpackNormal(tex2D(_NormalMap,i.uv));
                nDirTS = lerp(float3(0,0,1),nDirTS,_NormalIntensity);
            	
                half3x3 tangent2World = half3x3(i.tangentWS,i.binormalWS ,i.normalWS);
                half3 nDirWS = normalize(mul(nDirTS,tangent2World));
                
                half3 ViewN =  i.normalVS_uv;
                //ViewN.xy += nDirTS.xy * _Bump;
                
                half NdotL = dot(nDirWS,normalize(light.direction));
                NdotL = max(0,NdotL);
                half NdotV = dot(nDirWS,vDirWS);
                half NdotH = dot(nDirWS,normalize(light.direction + vDirWS));
                NdotH = max(0,NdotH);
                half halfLambert = smoothstep(0,0.5,NdotL);
                
                
                // sample the texture
                half4 mainTex  = tex2D(_MainTex, i.uv).rgba ;
                half4 lightMap = tex2D(_lightMap,i.uv);
                half ao = i.vertex_color.r;
            	
                //shadow范围
                half lambert_term = halfLambert * ao ; //shadow_control就是上面所说的阴影倾向权重
                half toon_diffuse = saturate((lambert_term - _ShadowThreshold) * _ShadowHardness);  //色阶化处理得到亮暗部分mask

            	//shadow颜色
                half rampLayer = lightMap.a;
				half tightsLayer = step(abs(rampLayer - 0.7), 0.015); //5
				half softLayer = (1.0 - tightsLayer) * step(abs(rampLayer - 0.3), 0.015);//4
				half metalLayer = (1.0 - softLayer) * step(abs(rampLayer - 0.5), 0.015); //3
				half skinLayer = (1.0 - metalLayer) * step(abs(rampLayer - 0.9), 0.015); //2
				half hardLayer = saturate(1.0 - (tightsLayer + softLayer + metalLayer + skinLayer));//1

				mainTex.xyz *= (_Color.xyz * hardLayer + _Color2.xyz * skinLayer + _Color3.xyz * metalLayer + _Color4.xyz * softLayer + _Color5 * tightsLayer);
				float3 shadowRamp = (_ShadowMultColor.xyz * hardLayer + _ShadowMultColor2.xyz * skinLayer + _ShadowMultColor3.xyz * metalLayer + _ShadowMultColor4.xyz * softLayer + _ShadowMultColor5.xyz * tightsLayer);
                
                

            	half brightScale = 1;
            	half brightArea = brightScale;
            	
            	half3 diffuseColor = lerp(shadowRamp * mainTex.xyz, mainTex.xyz, toon_diffuse);
                //高光点
                //half var_Meta = step(0.9,lightMap.r) ;
                half var_Meta = lightMap.r ;
                var_Meta = step(0.95,var_Meta);
                half var_Spec = lightMap.b ;
                // half HardSpec_term = smoothstep(0.1,1-_specHardness,lightMap.b * pow(NdotV,200* (1 - _SpecSize)) * step(0.95,lightMap.r) ) ;
                half3 Var_SpecRamp = tex2D(_RampTexSpec ,pow(NdotH,10) );
                half HardSpec_term = lightMap.b * Var_SpecRamp.r * step(0.95,lightMap.r)  ;
            	// half HardSpec_term2 = smoothstep(0.1,1-_specHardness2,lightMap.b * pow(NdotV,800* (1 - _SpecSize2 )) * step(0.95,lightMap.r) ) ;
            	half HardSpec_term2 = lightMap.b * Var_SpecRamp.g * step(0.95,lightMap.r)  ;
                // HardSpec_term2 = saturate(HardSpec_term2 * 10 );
                half HardSpec = HardSpec_term * _HardSpecInt  + HardSpec_term2 * _HardSpecInt2 ;
                HardSpec *= NdotL ;

                
                //YS高光
                half metalFact = tex2D(_MetalMap,ViewN).r;
                half3 SpecularColor = lerp(half3(0.3, 0.3, 0.3), diffuseColor,  0.1) * var_Spec;
                 metalFact *= SpecularColor * lerp(_MetalIntensity * 5,_MetalIntensity  * 10, var_Meta) ;
                half3 MetalSpecColor = metalFact * diffuseColor ;

            	//环境光提亮
                _env_specInt = 0.1;
                MetalSpecColor = max(MetalSpecColor,diffuseColor*_env_specInt)*var_Meta;
                
                half3 NPRMetaSpec = lerp(diffuseColor , MetalSpecColor, _MetalIntensity);//进行混合
            	

                //MatCap菲涅尔
                // half RimColor = tex2D(_RimMap,ViewN).r * _RimIntensity;
                
                
                //rimdark
                half rimdark_term = smoothstep(_SmoothstepMin,_SmoothstepMax,NdotV);
                rimdark_term = lerp( 1 ,rimdark_term, 1- toon_diffuse);
                half3 GBVSrim2 = rimdark_term * _RimLightColor;

                
                half4 finalcol = 1;
                finalcol.rgb = NPRMetaSpec  + HardSpec  +
                    diffuseColor * ( 1 - var_Meta) ;
                // finalcol.xyz = MetalSpecColor;

                return finalcol;
            }
            ENDHLSL
        }
    }
}
