import { AlertCircle, Clock, User, Database } from "lucide-react";
import { useEffect, useRef, useState } from "react";

const problems = [
  {
    icon: User,
    title: "Subjective Manual Sketching",
    description: "Human sketches vary widely based on artist interpretation, leading to inconsistent results.",
  },
  {
    icon: Clock,
    title: "Time-Consuming Investigations",
    description: "Traditional methods take days or weeks to produce and refine forensic sketches.",
  },
  {
    icon: AlertCircle,
    title: "Dependency on Human Accuracy",
    description: "Witness memory fades quickly, and manual processes introduce errors.",
  },
  {
    icon: Database,
    title: "Database Matching Inefficiencies",
    description: "Manual comparison across thousands of records is slow and prone to mistakes.",
  },
];

export const ProblemStatement = () => {
  const sectionRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.2 }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <section ref={sectionRef} className="py-24 px-6 relative overflow-hidden">
      <div className="container mx-auto">
        <div className={`text-center mb-16 transition-all duration-700 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            The Challenge in <span className="text-gradient">Criminal Identification</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Traditional forensic sketch methods face critical limitations that slow down justice.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {problems.map((problem, index) => {
            const Icon = problem.icon;
            return (
              <div
                key={index}
                className={`glass-card p-6 rounded-xl hover:glow-effect transition-all duration-500 hover:scale-105 ${
                  isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
                }`}
                style={{ transitionDelay: `${index * 100}ms` }}
              >
                <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center mb-4">
                  <Icon className="w-6 h-6 text-primary" />
                </div>
                <h3 className="text-lg font-semibold mb-2">{problem.title}</h3>
                <p className="text-sm text-muted-foreground">{problem.description}</p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};
