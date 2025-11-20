import { TrendingUp, Clock, Target, Users } from "lucide-react";

const metrics = [
  {
    icon: TrendingUp,
    value: "85%",
    label: "Faster Identification",
    description: "Compared to traditional manual sketch methods",
  },
  {
    icon: Clock,
    value: "< 5min",
    label: "Processing Time",
    description: "From description to database match",
  },
  {
    icon: Target,
    value: "92%",
    label: "Match Accuracy",
    description: "Validated across real-world cases",
  },
  {
    icon: Users,
    value: "1000+",
    label: "Cases Solved",
    description: "Law enforcement agencies worldwide",
  },
];

const testimonials = [
  {
    quote: "This system has revolutionized how we approach suspect identification. The speed and accuracy are unprecedented.",
    author: "Chief Detective, Metropolitan Police",
    role: "Law Enforcement",
  },
  {
    quote: "The AI-generated sketches match witness descriptions with remarkable precision. A game-changer for forensic investigations.",
    author: "Dr. Sarah Johnson",
    role: "Forensic Specialist",
  },
];

export const Impact = () => {
  return (
    <section className="py-24 px-6 bg-secondary/30">
      <div className="container mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Real-World <span className="text-gradient">Impact</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Transforming forensic investigations with measurable results.
          </p>
        </div>

        {/* Metrics */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
          {metrics.map((metric, index) => {
            const Icon = metric.icon;
            return (
              <div
                key={index}
                className="glass-card p-6 rounded-xl text-center hover:glow-effect transition-all duration-300"
              >
                <div className="w-12 h-12 mx-auto mb-4 rounded-lg bg-primary/20 flex items-center justify-center">
                  <Icon className="w-6 h-6 text-primary" />
                </div>
                <div className="text-4xl font-bold text-gradient mb-2">{metric.value}</div>
                <div className="font-semibold mb-1">{metric.label}</div>
                <p className="text-sm text-muted-foreground">{metric.description}</p>
              </div>
            );
          })}
        </div>

        {/* Testimonials */}
        <div className="max-w-4xl mx-auto">
          <h3 className="text-2xl font-semibold text-center mb-8">Trusted by Professionals</h3>
          <div className="grid md:grid-cols-2 gap-6">
            {testimonials.map((testimonial, index) => (
              <div key={index} className="glass-card p-6 rounded-xl">
                <div className="mb-4">
                  <svg
                    className="w-8 h-8 text-primary/50"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path d="M14.017 21v-7.391c0-5.704 3.731-9.57 8.983-10.609l.995 2.151c-2.432.917-3.995 3.638-3.995 5.849h4v10h-9.983zm-14.017 0v-7.391c0-5.704 3.748-9.57 9-10.609l.996 2.151c-2.433.917-3.996 3.638-3.996 5.849h3.983v10h-9.983z" />
                  </svg>
                </div>
                <p className="text-lg mb-4 leading-relaxed">{testimonial.quote}</p>
                <div>
                  <p className="font-semibold">{testimonial.author}</p>
                  <p className="text-sm text-primary">{testimonial.role}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Applications */}
        <div className="mt-16 text-center">
          <h3 className="text-2xl font-semibold mb-8">Applications</h3>
          <div className="flex flex-wrap justify-center gap-3">
            {[
              "Criminal Investigations",
              "Missing Persons",
              "Cold Case Resolution",
              "Border Security",
              "Event Security",
              "Witness Protection",
            ].map((app, index) => (
              <div
                key={index}
                className="glass-card px-4 py-2 rounded-full text-sm hover:glow-effect transition-all cursor-default"
              >
                {app}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};
