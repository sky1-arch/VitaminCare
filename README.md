# VitaminCare

A React + Vite version of the VitaminCare screening demo, prepared for GitHub/GitHub Pages.

## Run locally

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

The app uses browser `localStorage`, so accounts and assessment reports are stored in the browser on that device/browser. This is suitable for a demo, not production medical data.

## GitHub Pages

1. Upload this project to a GitHub repository.
2. In GitHub, open **Settings → Pages**.
3. Choose **GitHub Actions** as the deployment source.
4. Add a workflow that runs `npm ci`, `npm run build`, and deploys the `dist` folder.

The Vite `base: "./"` setting makes the build work when hosted under a GitHub Pages repository path.
