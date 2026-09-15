const path = require('path');

module.exports = {
  // ── Disable the CRA ESLint webpack plugin entirely ──────────────────────
  // Fixes: "Failed to load plugin 'jest' ... Cannot read properties of undefined"
  eslint: {
    enable: false,
  },

  webpack: {
    configure: (webpackConfig) => {

      // ── Suppress source-map-loader warnings from node_modules ────────────
      // Fixes: "Failed to parse source map from vision_bundle_mjs.js.map"
      webpackConfig.ignoreWarnings = [
        ...(webpackConfig.ignoreWarnings || []),
        /Failed to parse source map/,
      ];

      // ── Suppress "Critical dependency: the request of a dependency is an expression" ──
      // This comes from @mediapipe/tasks-vision's internal dynamic require inside vision_bundle.mjs
      webpackConfig.module = webpackConfig.module || {};
      webpackConfig.module.exprContextCritical = false;

      // Also add a specific no-parse rule so webpack stops trying to analyse
      // the MediaPipe WASM bundle's dynamic expressions
      const mediapipePath = path.resolve(
        __dirname,
        'node_modules/@mediapipe/tasks-vision'
      );

      webpackConfig.module.rules = [
        ...(webpackConfig.module.rules || []),
        {
          test: /vision_bundle\.mjs$/,
          include: mediapipePath,
          type: 'javascript/auto',
          resolve: { fullySpecified: false },
        },
      ];

      // Fix dynamic import() in mediapipe for production build
      webpackConfig.output = webpackConfig.output || {};
      webpackConfig.output.environment = {
        ...webpackConfig.output.environment,
        dynamicImport: true,
      };

      // Suppress source-map-loader from processing node_modules at all
      webpackConfig.module.rules = webpackConfig.module.rules.map((rule) => {
        if (rule.enforce === 'pre' && rule.use) {
          const uses = Array.isArray(rule.use) ? rule.use : [rule.use];
          const hasSourceMapLoader = uses.some(
            (u) => (typeof u === 'string' ? u : u.loader || '')
              .includes('source-map-loader')
          );
          if (hasSourceMapLoader) {
            return {
              ...rule,
              exclude: /node_modules/,
            };
          }
        }
        return rule;
      });

      return webpackConfig;
    },
  },
  devServer: {
    allowedHosts: 'all',
  },
};
