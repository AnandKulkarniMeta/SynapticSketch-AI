import { Brain, Palette, Edit, Scan, Zap, Target } from "lucide-react";

const features = [
  {
    icon: Brain,
    title: "NLP-Driven Interpretation",
    description: "Advanced natural language models extract precise facial features from witness descriptions.",
    gradient: "from-blue-500 to-cyan-500",
  },
  {
    icon: Palette,
    title: "Diffusion + GAN Generator",
    description: "State-of-the-art AI creates photorealistic forensic sketches with exceptional detail.",
    gradient: "from-cyan-500 to-teal-500",
  },
  {
    icon: Edit,
    title: "Live Sketch Editing",
    description: "Interactive interface for real-time refinement based on additional witness feedback.",
    gradient: "from-teal-500 to-green-500",
  },
  {
    icon: Scan,
    title: "Deep Face Recognition",
    description: "ArcFace and FaceNet embeddings ensure highly accurate facial feature mapping.",
    gradient: "from-green-500 to-emerald-500",
  },
  {
    icon: Zap,
    title: "High-Speed Matching",
    description: "Process thousands of database records in seconds with optimized search algorithms.",
    gradient: "from-emerald-500 to-cyan-500",
  },
  {
    icon: Target,
    title: "Forensic-Grade Accuracy",
    description: "Validated against real-world cases with exceptional identification success rates.",
    gradient: "from-cyan-500 to-blue-500",
  },
];

export const Features = () => {
  return (
    <section className="py-24 px-6 bg-secondary/30">
      <div className="container mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Key <span className="text-gradient">Features</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Cutting-edge AI capabilities designed for modern forensic investigations.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <div
                key={index}
                className="group glass-card p-6 rounded-xl hover:glow-effect transition-all duration-300 hover:-translate-y-2"
              >
                <div className="relative mb-4">
                  <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${feature.gradient} opacity-20 group-hover:opacity-30 transition-opacity`} />
                  <Icon className="w-7 h-7 text-primary absolute top-3.5 left-3.5" />
                </div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};
