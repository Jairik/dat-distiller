import { AnimatedNumber, HoverPress, PageIn, StaggerGroup, StaggerItem } from '@/components/motion'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

const STEPS = ['Generate', 'Label', 'Train'] as const

export default function App() {
  return (
    <PageIn className="mx-auto flex min-h-svh w-full max-w-2xl flex-col justify-center gap-8 p-8">
      <header className="flex items-center gap-3">
        <h1 className="text-4xl font-bold tracking-tight">Dat Distiller</h1>
        <Badge className="bg-primary/15 text-primary dark:bg-primary/20">local workbench</Badge>
      </header>
      <p className="-mt-4 text-muted-foreground">
        Generate synthetic data, label it with Jev, and train models — all on your own machine.
      </p>
      <Card>
        <CardHeader>
          <CardTitle>Three steps, any order</CardTitle>
          <CardDescription>Each step can start from the previous one or from an upload.</CardDescription>
        </CardHeader>
        <CardContent className="flex items-center justify-between gap-8">
          <StaggerGroup className="grid gap-2">
            {STEPS.map((step) => (
              <StaggerItem key={step}>
                <Button variant="secondary" className="w-40 justify-start">
                  {step}
                </Button>
              </StaggerItem>
            ))}
          </StaggerGroup>
          <div className="text-right">
            <AnimatedNumber value={3} className="text-6xl font-semibold text-primary" />
            <p className="text-sm text-muted-foreground">steps to a model</p>
          </div>
        </CardContent>
      </Card>
      <div>
        <HoverPress>
          <Button size="lg">Start a Project</Button>
        </HoverPress>
      </div>
    </PageIn>
  )
}
