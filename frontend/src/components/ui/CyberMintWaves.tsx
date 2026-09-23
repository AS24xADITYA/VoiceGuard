import React, { useEffect, useRef } from 'react';

interface Wave3DLayer {
  zDepth: number; // Simulated Z distance (0 = foreground, 280 = deep distance)
  baseYRatio: number; // Base vertical anchor (relative to container height)
  amplitudes: [number, number, number]; // Harmonic amplitudes
  frequencies: [number, number, number]; // Incommensurate spatial frequencies
  speeds: [number, number, number]; // Temporal speeds
  fillGrad: (ctx: CanvasRenderingContext2D, w: number, h: number, scale: number) => CanvasGradient;
  strokeColor: string;
  glowColor: string;
  glowBlur: number;
  lineWidth: number;
}

export const CyberMintWaves: React.FC<{ className?: string }> = ({ className = '' }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) return;

    let animationFrameId: number;
    let logicalWidth = container.clientWidth || window.innerWidth;
    let logicalHeight = container.clientHeight || 520;

    // Interactive Parallax tracking with smooth linear interpolation
    let targetParallaxX = 0;
    let targetParallaxY = 0;
    let currentParallaxX = 0;
    let currentParallaxY = 0;

    const onPointerMove = (e: PointerEvent) => {
      const normX = (e.clientX / window.innerWidth) * 2 - 1; // -1 to +1
      const normY = (e.clientY / window.innerHeight) * 2 - 1;
      targetParallaxX = normX * 35;
      targetParallaxY = normY * 12;
    };

    window.addEventListener('pointermove', onPointerMove, { passive: true });

    // Accessibility: prefers-reduced-motion
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');

    // 6 Stratified 3D perspective layers with 1.5x vertical range
    const focalLength = 500;
    const layers: Wave3DLayer[] = [
      // Layer 1: Distant stratosphere wave (Z = 440) — new upper wave line 1
      {
        zDepth: 440,
        baseYRatio: 0.63,
        amplitudes: [15, 9, 5],
        frequencies: [0.0013, 0.0026, 0.0046],
        speeds: [-0.00018, 0.00013, -0.00020],
        fillGrad: (c, _w, h) => {
          const g = c.createLinearGradient(0, h * 0.48, 0, h);
          g.addColorStop(0, 'rgba(6, 78, 59, 0.14)');
          g.addColorStop(0.5, 'rgba(6, 38, 30, 0.40)');
          g.addColorStop(1, 'rgba(6, 11, 10, 0.92)');
          return g;
        },
        strokeColor: 'rgba(16, 185, 129, 0.28)',
        glowColor: 'rgba(16, 185, 129, 0.15)',
        glowBlur: 6,
        lineWidth: 1.0,
      },
      // Layer 2: High horizon dune (Z = 360) — new upper wave line 2
      {
        zDepth: 360,
        baseYRatio: 0.69,
        amplitudes: [18, 11, 6],
        frequencies: [0.0011, 0.0023, 0.0041],
        speeds: [0.00020, -0.00014, 0.00022],
        fillGrad: (c, _w, h) => {
          const g = c.createLinearGradient(0, h * 0.54, 0, h);
          g.addColorStop(0, 'rgba(6, 78, 59, 0.18)');
          g.addColorStop(0.5, 'rgba(6, 38, 30, 0.48)');
          g.addColorStop(1, 'rgba(6, 11, 10, 0.94)');
          return g;
        },
        strokeColor: 'rgba(16, 185, 129, 0.35)',
        glowColor: 'rgba(16, 185, 129, 0.20)',
        glowBlur: 7,
        lineWidth: 1.1,
      },
      // Layer 3: Deep horizon dune (Z = 280)
      {
        zDepth: 280,
        baseYRatio: 0.74,
        amplitudes: [20, 12, 6],
        frequencies: [0.0012, 0.0024, 0.0042],
        speeds: [0.00022, -0.00015, 0.00025],
        fillGrad: (c, _w, h) => {
          const g = c.createLinearGradient(0, h * 0.6, 0, h);
          g.addColorStop(0, 'rgba(6, 78, 59, 0.20)');
          g.addColorStop(0.5, 'rgba(6, 38, 30, 0.55)');
          g.addColorStop(1, 'rgba(6, 11, 10, 0.95)');
          return g;
        },
        strokeColor: 'rgba(16, 185, 129, 0.40)',
        glowColor: 'rgba(16, 185, 129, 0.24)',
        glowBlur: 8,
        lineWidth: 1.2,
      },
      // Layer 4: Mid-distance undulating terrain (Z = 180)
      {
        zDepth: 180,
        baseYRatio: 0.80,
        amplitudes: [24, 14, 8],
        frequencies: [0.0010, 0.0020, 0.0036],
        speeds: [-0.00026, 0.00020, -0.00016],
        fillGrad: (c, _w, h) => {
          const g = c.createLinearGradient(0, h * 0.66, 0, h);
          g.addColorStop(0, 'rgba(16, 185, 129, 0.24)');
          g.addColorStop(0.4, 'rgba(8, 45, 35, 0.65)');
          g.addColorStop(1, 'rgba(6, 11, 10, 0.96)');
          return g;
        },
        strokeColor: 'rgba(52, 211, 153, 0.55)',
        glowColor: 'rgba(16, 185, 129, 0.30)',
        glowBlur: 12,
        lineWidth: 1.5,
      },
      // Layer 5: Near-ground resonant wave (Z = 90)
      {
        zDepth: 90,
        baseYRatio: 0.86,
        amplitudes: [28, 16, 10],
        frequencies: [0.0009, 0.0018, 0.0032],
        speeds: [0.00032, -0.00024, 0.00018],
        fillGrad: (c, _w, h) => {
          const g = c.createLinearGradient(0, h * 0.72, 0, h);
          g.addColorStop(0, 'rgba(16, 185, 129, 0.30)');
          g.addColorStop(0.35, 'rgba(6, 42, 33, 0.75)');
          g.addColorStop(1, 'rgba(6, 11, 10, 0.98)');
          return g;
        },
        strokeColor: 'rgba(16, 185, 129, 0.70)',
        glowColor: 'rgba(16, 185, 129, 0.45)',
        glowBlur: 16,
        lineWidth: 1.8,
      },
      // Layer 6: Foreground luminous crest (Z = 10)
      {
        zDepth: 10,
        baseYRatio: 0.92,
        amplitudes: [32, 18, 10],
        frequencies: [0.0011, 0.0022, 0.0038],
        speeds: [-0.00040, 0.00030, -0.00022],
        fillGrad: (c, _w, h) => {
          const g = c.createLinearGradient(0, h * 0.80, 0, h);
          g.addColorStop(0, 'rgba(52, 211, 153, 0.35)');
          g.addColorStop(0.3, 'rgba(8, 48, 38, 0.82)');
          g.addColorStop(1, 'rgba(6, 11, 10, 1.0)');
          return g;
        },
        strokeColor: 'rgba(167, 243, 208, 0.88)',
        glowColor: 'rgba(52, 211, 153, 0.65)',
        glowBlur: 20,
        lineWidth: 2.2,
      },
    ];

    // High-DPI canvas buffer resizing function anchored strictly to the hero container
    const resizeBuffer = () => {
      if (!container || !canvas) return;
      const rect = container.getBoundingClientRect();
      logicalWidth = rect.width || window.innerWidth;
      logicalHeight = rect.height || 520;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(logicalWidth * dpr);
      canvas.height = Math.floor(logicalHeight * dpr);

      // Reset transform and scale to DPI
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const resizeObserver = new ResizeObserver(() => {
      resizeBuffer();
      if (motionQuery.matches) {
        renderFrame(1000);
      }
    });
    resizeObserver.observe(container);
    window.addEventListener('resize', resizeBuffer);

    // Initial buffer setup
    resizeBuffer();

    const draw3DWave = (time: number, layer: Wave3DLayer) => {
      const scale = focalLength / (focalLength + layer.zDepth);

      const parallaxFactor = 1 - layer.zDepth / 400;
      const offsetX = currentParallaxX * parallaxFactor;
      const offsetY = currentParallaxY * parallaxFactor;

      const baseY = logicalHeight * layer.baseYRatio + offsetY;

      ctx.beginPath();
      ctx.moveTo(0, logicalHeight);

      const step = 4;
      for (let x = -20; x <= logicalWidth + 20 + step; x += step) {
        const sampleX = x - offsetX;
        const waveY =
          (Math.sin(sampleX * layer.frequencies[0] + time * layer.speeds[0]) * layer.amplitudes[0] +
            Math.sin(sampleX * layer.frequencies[1] + time * layer.speeds[1]) * layer.amplitudes[1] +
            Math.cos(sampleX * layer.frequencies[2] + time * layer.speeds[2]) * layer.amplitudes[2]) *
          scale;

        const y = baseY + waveY;
        if (x === -20) {
          ctx.lineTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.lineTo(logicalWidth + 20, logicalHeight);
      ctx.closePath();

      ctx.fillStyle = layer.fillGrad(ctx, logicalWidth, logicalHeight, scale);
      ctx.fill();

      // Stroke luminous 3D crest
      ctx.save();
      ctx.strokeStyle = layer.strokeColor;
      ctx.lineWidth = layer.lineWidth * scale;
      ctx.shadowColor = layer.glowColor;
      ctx.shadowBlur = layer.glowBlur * scale;
      ctx.stroke();
      ctx.restore();
    };

    const renderFrame = (timestamp: number) => {
      ctx.clearRect(0, 0, logicalWidth, logicalHeight);

      // Smooth parallax interpolation toward target
      currentParallaxX += (targetParallaxX - currentParallaxX) * 0.04;
      currentParallaxY += (targetParallaxY - currentParallaxY) * 0.04;

      // Autonomous subtle breathing motion when idle
      const idleTime = timestamp * 0.001;
      const idleOffsetX = Math.sin(idleTime * 0.5) * 6;
      const idleOffsetY = Math.cos(idleTime * 0.4) * 4;
      currentParallaxX += idleOffsetX * 0.02;
      currentParallaxY += idleOffsetY * 0.02;

      for (let i = 0; i < layers.length; i++) {
        draw3DWave(timestamp, layers[i]);
      }
    };

    if (motionQuery.matches) {
      renderFrame(1000);
      return () => {
        resizeObserver.disconnect();
        window.removeEventListener('resize', resizeBuffer);
        window.removeEventListener('pointermove', onPointerMove);
      };
    }

    const loop = (timestamp: number) => {
      renderFrame(timestamp);
      animationFrameId = requestAnimationFrame(loop);
    };

    animationFrameId = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      window.removeEventListener('resize', resizeBuffer);
      window.removeEventListener('pointermove', onPointerMove);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className={`absolute inset-0 pointer-events-none z-0 overflow-hidden ${className}`}
    >
      <canvas
        ref={canvasRef}
        className="block w-full h-full"
      />
      {/* Soft gradient mask: ensures waves stay behind text and fade gently into the page base */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            'linear-gradient(to bottom, #060B0A 0%, rgba(6, 11, 10, 0.82) 20%, rgba(6, 11, 10, 0.15) 44%, transparent 58%, transparent 84%, #060B0A 100%)',
        }}
      />
    </div>
  );
};
