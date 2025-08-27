import ResultPageClient from './ResultPageClient';

interface PageProps {
  params: Promise<{ resultId: string }>;
}

export const dynamic = 'force-static';
export const revalidate = 0;

export default async function ResultPage({ params }: PageProps) {
  const { resultId } = await params;
  return <ResultPageClient resultId={resultId} />;
}

export async function generateStaticParams() {
  // No pre-rendered result pages; this keeps static export happy without generating any.
  return [] as Array<{ resultId: string }>;
}
