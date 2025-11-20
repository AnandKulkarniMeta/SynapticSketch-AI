import { MessageSquare, Wand2, Edit3, Fingerprint, Search } from "lucide-react";
import { useEffect, useRef, useState } from "react";

const steps = [
  {
    icon: MessageSquare,
    title: "NLP Extraction",
    description: "Advanced natural language processing interprets witness statements to extract detailed facial features.",
    step: "01",
  },
  {
    icon: Wand2,
    title: "AI Sketch Generation",
    description: "Diffusion models and GANs collaborate to create highly accurate forensic sketches from descriptions.",
    step: "02",
  },
  {
    icon: Edit3,
    title: "Interactive Refinement",
    description: "Live editing interface allows investigators to fine-tune sketches based on additional witness feedback.",
    step: "03",
  },
  {
    icon: Fingerprint,
    title: "Face Embeddings",
    description: "ArcFace and FaceNet create precise mathematical representations for accurate matching.",
    step: "04",
  },
  {
    icon: Search,
    title: "Database Matching",
    description: "Lightning-fast comparison across criminal databases to identify potential suspects instantly.",
    step: "05",
  },
];

export const Solution = () => {
  const sectionRef = useRef<HTMLDivElement>(null);
  const [visibleSteps, setVisibleSteps] = useState<number[]>([]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const index = parseInt(entry.target.getAttribute("data-index") || "0");
            setVisibleSteps((prev) => [...new Set([...prev, index])]);
          }
        });
      },
      { threshold: 0.3 }
    );

    const elements = sectionRef.current?.querySelectorAll(".solution-step");
    elements?.forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, []);

  return (
    <section ref={sectionRef} className="py-24 px-6 bg-secondary/30">
      <div className="container mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            AI-Powered <span className="text-gradient">Solution</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            A seamless pipeline from witness statement to suspect identification.
          </p>
        </div>

        <div className="relative max-w-5xl mx-auto">
          {/* Connection line */}
          <div className="hidden lg:block absolute left-1/2 top-0 bottom-0 w-0.5 bg-gradient-to-b from-primary/0 via-primary/50 to-primary/0" />

          <div className="space-y-12">
            {steps.map((step, index) => {
              const Icon = step.icon;
              const isVisible = visibleSteps.includes(index);
              const isLeft = index % 2 === 0;

              return (
                <div
                  key={index}
                  data-index={index}
                  className={`solution-step flex items-center gap-8 ${
                    isLeft ? "lg:flex-row" : "lg:flex-row-reverse"
                  }`}
                >
                  <div
                    className={`flex-1 transition-all duration-700 ${
                      isVisible
                        ? "opacity-100 translate-x-0"
                        : `opacity-0 ${isLeft ? "-translate-x-10" : "translate-x-10"}`
                    }`}
                  >
                    <div className="glass-card p-6 rounded-xl hover:glow-effect transition-all duration-300">
                      <div className="flex items-start gap-4">
                        <div className="w-14 h-14 rounded-lg bg-primary/20 flex items-center justify-center flex-shrink-0">
                          <Icon className="w-7 h-7 text-primary" />
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <span className="text-xs font-mono text-primary">{step.step}</span>
                            <h3 className="text-xl font-semibold">{step.title}</h3>
                          </div>
                          <p className="text-muted-foreground">{step.description}</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="hidden lg:block w-12 h-12 rounded-full glass-card border-2 border-primary/50 flex-shrink-0 relative z-10">
                    <div
                      className={`absolute inset-0 rounded-full bg-primary/30 transition-all duration-700 ${
                        isVisible ? "scale-100 opacity-100" : "scale-0 opacity-0"
                      }`}
                    />
                  </div>

                  <div className="flex-1 hidden lg:block" />
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
};
