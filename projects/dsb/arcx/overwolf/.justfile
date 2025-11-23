# Overwolf-specific justfile
# Can be used from overwolf directory: just -f .justfile <recipe>

# Install dependencies
install:
    pnpm install

# Build TypeScript
build:
    pnpm run build

# Build in watch mode
watch:
    pnpm run watch

# Clean build artifacts
clean:
    rm -rf dist/ node_modules/

# Reinstall dependencies
reinstall: clean install

# Check TypeScript
check:
    pnpm exec tsc --noEmit

# Show compiled JS
show-js:
    @ls -lh dist/*.js 2>/dev/null || echo "No built files"

# Update dependencies
update:
    pnpm update

# Show outdated packages
outdated:
    pnpm outdated
