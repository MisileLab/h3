# Frontend AI Agent Guidelines

## Tech Stack
- **React** with TypeScript
- **Tailwind CSS** for styling
- **shadcn/ui** components
- **React Router** for navigation

## Key Requirements

### Code Standards
- Use TypeScript interfaces for all props
- Implement proper error handling and loading states
- Follow mobile-first responsive design
- Use shadcn/ui components where available
- Ensure WCAG 2.1 accessibility compliance

### Project Structure
```
src/
├── components/ui/     # shadcn/ui components
├── pages/            # Route components
├── hooks/            # Custom hooks
├── lib/              # Utils & config
└── types/            # TS definitions
```

### Implementation Patterns
- **Components**: Functional with hooks, proper TypeScript, React.memo for performance
- **Routing**: Nested routes, protected routes, proper error pages
- **Styling**: Tailwind utilities only, shadcn/ui theme system
- **State**: Built-in React state management (useState, useContext, custom hooks)
- **Forms**: Controlled components with validation using shadcn/ui Form components
- **Data**: Custom hooks for API calls, proper loading/error states

### Response Format
Always provide:
- Complete working code with TypeScript
- Proper imports and file structure
- Usage examples
- Error boundaries where needed
