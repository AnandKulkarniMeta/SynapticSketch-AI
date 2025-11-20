import { Button } from "@/components/ui/button";
import { Scan, Sparkles } from "lucide-react";
import { useEffect, useRef } from "react";

export const Hero = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles: Array<{
      x: number;
      y: number;
      vx: number;
      vy: number;
      size: number;
    }> = [];

    // Create particles
    for (let i = 0; i < 50; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        size: Math.random() * 2 + 1,
      });
    }

    const animate = () => {
      ctx.fillStyle = "rgba(17, 24, 39, 0.1)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      particles.forEach((particle, i) => {
        particle.x += particle.vx;
        particle.y += particle.vy;

        if (particle.x < 0 || particle.x > canvas.width) particle.vx *= -1;
        if (particle.y < 0 || particle.y > canvas.height) particle.vy *= -1;

        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0, 212, 255, 0.6)";
        ctx.fill();

        // Draw connections
        particles.slice(i + 1).forEach((otherParticle) => {
          const dx = particle.x - otherParticle.x;
          const dy = particle.y - otherParticle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 150) {
            ctx.beginPath();
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(otherParticle.x, otherParticle.y);
            ctx.strokeStyle = `rgba(0, 212, 255, ${0.2 * (1 - distance / 150)})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        });
      });

      requestAnimationFrame(animate);
    };

    animate();

    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      <canvas
        ref={canvasRef}
        className="absolute inset-0 z-0"
        style={{ background: "linear-gradient(180deg, hsl(220 30% 8%) 0%, hsl(220 30% 4%) 100%)" }}
      />
      
      {/* Scan line effect */}
      <div className="absolute inset-0 z-10 pointer-events-none">
        <div className="h-0.5 w-full bg-gradient-to-r from-transparent via-primary to-transparent opacity-20 animate-scan-line" />
      </div>

      <div className="container mx-auto px-6 z-20 text-center">
        <div className="space-y-6 animate-fade-in-up">
          <div className="inline-flex items-center gap-2 px-4 py-2 glass-card rounded-full mb-4">
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="text-sm text-muted-foreground">Powered by Advanced AI</span>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold leading-tight">
            Reinventing Criminal Identification
            <br />
            <span className="text-gradient glow-text">with AI</span>
          </h1>
          
          <p className="text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto">
            Automated forensic sketch generation and intelligent suspect matching.
            <br />
            Transform witness descriptions into actionable intelligence.
          </p>

          <div className="flex gap-4 justify-center pt-8">
            <Button size="lg" className="gap-2 glow-effect hover:scale-105 transition-transform">
              <Scan className="w-5 h-5" />
              Try Demo
            </Button>
            <Button size="lg" variant="outline" className="gap-2 glass-card hover:glow-effect hover:scale-105 transition-transform">
              Explore Features
            </Button>
          </div>
        </div>

        {/* Face transformation visual placeholder */}
        <div className="mt-16 relative max-w-4xl mx-auto animate-float">
          <div className="glass-card rounded-2xl p-8 glow-effect">
            <div className="grid md:grid-cols-3 gap-8 items-center">
              <div className="text-center">
                <div className="w-32 h-32 mx-auto rounded-full bg-secondary/50 border-2 border-primary/30 flex items-center justify-center mb-3">
                  <span className="text-4xl">👤</span>
                </div>
                <p className="text-sm text-muted-foreground">Witness Description</p>
              </div>
              
              <div className="hidden md:block text-center">
                <div className="text-primary text-2xl">→ AI Processing →</div>
              </div>
              
              <div className="text-center">
                <div className="w-32 h-32 mx-auto rounded-full bg-secondary/50 border-2 border-primary/30 flex items-center justify-center mb-3 animate-glow-pulse">
                  <Scan className="w-12 h-12 text-primary" />
                </div>
                <p className="text-sm text-muted-foreground">Forensic Sketch</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
