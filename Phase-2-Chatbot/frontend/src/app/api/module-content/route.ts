import { NextRequest } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const moduleId = searchParams.get('id');
    const topicId = searchParams.get('topic'); // Optional parameter for specific topic

    if (!moduleId) {
      return Response.json({ error: 'Module ID is required' }, { status: 400 });
    }

    // Path to the docs directory - since we're in frontend/src/app/api/
    // According to user instructions, docs folder is in frontend/src
    const docsPath = path.join(process.cwd(), 'src', 'docs');

    if (topicId) {
      // Handle specific topic request: /api/module-content?id=module1&topic=rclpy-agents
      const topicFileName = topicId.replace(/_/g, '-'); // Convert underscores to hyphens to match file naming
      const topicPath = path.join(docsPath, moduleId, `${topicFileName}.md`);

      if (fs.existsSync(topicPath)) {
        const content = fs.readFileSync(topicPath, 'utf8');
        return Response.json({ content });
      } else {
        // If specific topic file doesn't exist, return error
        return Response.json({ error: `Topic '${topicId}' not found in module '${moduleId}'` }, { status: 404 });
      }
    } else {
      // Handle general module request: /api/module-content?id=module1
      const modulePath = path.join(docsPath, moduleId, 'index.md');

      // If the specific module file doesn't exist, try looking for a generic module file
      if (!fs.existsSync(modulePath)) {
        // Try to find any markdown file in the module directory
        const moduleDirPath = path.join(docsPath, moduleId);
        if (fs.existsSync(moduleDirPath)) {
          const files = fs.readdirSync(moduleDirPath);
          const markdownFile = files.find(file => file.endsWith('.md'));
          if (markdownFile) {
            const fallbackPath = path.join(moduleDirPath, markdownFile);
            const content = fs.readFileSync(fallbackPath, 'utf8');
            return Response.json({ content });
          }
        }

        // If no markdown file is found in the specific module directory,
        // return a default content or look for module-specific files
        const introPath = path.join(docsPath, 'intro.md');
        if (fs.existsSync(introPath)) {
          const content = fs.readFileSync(introPath, 'utf8');
          return Response.json({ content });
        }

        return Response.json({ error: 'Module not found' }, { status: 404 });
      }

      const content = fs.readFileSync(modulePath, 'utf8');

      return Response.json({ content });
    }
  } catch (error) {
    console.error('Error reading module content:', error);
    return Response.json({ error: 'Internal server error' }, { status: 500 });
  }
}