import { useEffect, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { useFluidSimulation } from './useFluidSimulation'

const BG_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() { vUv = position.xy * 0.5 + 0.5; gl_Position = vec4(position.xy, 0.0, 1.0); }
`
const BG_FRAG = /* glsl */ `
  precision highp float;
  uniform sampler2D uFluid; varying vec2 vUv;
  void main() {
    vec3 d = max(texture2D(uFluid, vUv).rgb, 0.0);
    float l = dot(d, vec3(0.299, 0.587, 0.114));
    // HDR core: overlapping bright streaks bloom toward white
    vec3 c = d * 1.6 + d * l * 2.5 + vec3(pow(l, 2.0) * 1.2);
    c = 1.0 - exp(-c * 1.3);
    gl_FragColor = vec4(c, 1.0);
  }
`

type Getter = () => THREE.Texture

function FluidBackground({ getDensity }: { getDensity: Getter }) {
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3))
    return g
  }, [])
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: BG_VERT,
        fragmentShader: BG_FRAG,
        uniforms: { uFluid: { value: null as THREE.Texture | null } },
        depthTest: false,
        depthWrite: false,
      }),
    [],
  )
  useEffect(
    () => () => {
      geometry.dispose()
      material.dispose()
    },
    [geometry, material],
  )
  useFrame(() => {
    material.uniforms.uFluid.value = getDensity()
  })
  return <mesh geometry={geometry} material={material} renderOrder={-1} frustumCulled={false} />
}

function Scene() {
  const { getDensity } = useFluidSimulation()
  return (
    <>
      <color attach="background" args={['#000000']} />
      <FluidBackground getDensity={getDensity} />
    </>
  )
}

export default function FluidHeroScene() {
  return (
    <Canvas
      dpr={[1, 2]}
      camera={{ position: [0, 0, 4.5], fov: 40 }}
      gl={{ antialias: true, alpha: false }}
      style={{ position: 'fixed', inset: 0, zIndex: 0, background: '#000000', pointerEvents: 'none' }}
    >
      <Scene />
    </Canvas>
  )
}
