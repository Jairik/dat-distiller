/** Placeholder for project sections whose real page lands in a later issue. */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

export function SectionPlaceholder({ title, blurb }: { title: string; blurb: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{blurb}</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground">
          This section arrives with its feature issue. The app shell is wired so later
          issues only need to drop the real page in here.
        </p>
      </CardContent>
    </Card>
  )
}
