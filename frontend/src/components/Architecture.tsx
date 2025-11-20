import { useState } from "react";
import { Database, Brain, Image, Fingerprint, Search, ArrowRight } from "lucide-react";

const modules = [
  {
    id: "input",
    icon: Brain,
    title: "NLP Module",
    description: "Process witness statements",
  },
  {
    id: "generation",
    icon: Image,
    title: "Sketch Generator",
    description: "Diffusion + GAN models",
  },
  {
    id: "embedding",
    icon: Fingerprint,
    title: "Face Embeddings",
    description: "ArcFace & FaceNet",
  },
  {
    id: "database",
    icon: Database,
    title: "Criminal Database",
    description: "Secure data storage",
  },
  {
    id: "matching",
    icon: Search,
    title: "Matching Engine",
    description: "Similarity algorithms",
  },
];

export const Architecture = () => {
  const [hoveredModule, setHoveredModule] = useState<string | null>(null);

  return (
    <section className="py-24 px-6">
      <div className="container mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            System <span className="text-gradient">Architecture</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            An intelligent pipeline designed for accuracy, speed, and scalability.
          </p>
        </div>

        <div className="max-w-5xl mx-auto">
          <div className="glass-card rounded-2xl p-8">
            <div className="flex flex-wrap justify-center items-center gap-4">
              {modules.map((module, index) => {
                const Icon = module.icon;
                const isHovered = hoveredModule === module.id;

                return (
                  <div key={module.id} className="flex items-center">
                    <div
                      onMouseEnter={() => setHoveredModule(module.id)}
                      onMouseLeave={() => setHoveredModule(null)}
                      className={`relative transition-all duration-300 ${
                        isHovered ? "scale-110" : "scale-100"
                      }`}
                    >
                      <div
                        className={`glass-card p-6 rounded-xl cursor-pointer transition-all ${
                          isHovered ? "glow-effect" : ""
                        }`}
                      >
                        <div className="flex flex-col items-center gap-3 min-w-[140px]">
                          <div
                            className={`w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center transition-all ${
                              isHovered ? "bg-primary/30 scale-110" : ""
                            }`}
                          >
                            <Icon className="w-6 h-6 text-primary" />
                          </div>
                          <div className="text-center">
                            <h4 className="font-semibold text-sm mb-1">{module.title}</h4>
                            <p className="text-xs text-muted-foreground">{module.description}</p>
                          </div>
                        </div>
                      </div>

                      {isHovered && (
                        <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-0.5 h-8 bg-primary/50 animate-pulse" />
                      )}
                    </div>

                    {index < modules.length - 1 && (
                      <ArrowRight
                        className={`w-6 h-6 mx-2 transition-colors ${
                          isHovered || hoveredModule === modules[index + 1]?.id
                            ? "text-primary"
                            : "text-muted-foreground"
                        }`}
                      />
                    )}
                  </div>
                );
              })}
            </div>

            <div className="mt-8 text-center">
              <p className="text-sm text-muted-foreground">
                Hover over each module to see its role in the identification pipeline
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
