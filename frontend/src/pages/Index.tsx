import { Hero } from "@/components/Hero";
import { ProblemStatement } from "@/components/ProblemStatement";
import { Solution } from "@/components/Solution";
import { DemoPreview } from "@/components/DemoPreview";
import { Features } from "@/components/Features";
import { Architecture } from "@/components/Architecture";
import { Impact } from "@/components/Impact";
import { Footer } from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background text-foreground smooth-scroll">
      <Hero />
      <ProblemStatement />
      <Solution />
      <DemoPreview />
      <Features />
      <Architecture />
      <Impact />
      <Footer />
    </div>
  );
};

export default Index;
