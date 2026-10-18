import { defineConfig } from "vitepress";

export default defineConfig({
  lang: "en-US",
  title: "TokenSpeed PKU",
  description:
    "TokenSpeed course fork: Ascend experiments, evidence, and runtime documentation.",
  base: "/tokenspeed_PKU/",
  srcDir: ".",
  cleanUrls: true,
  lastUpdated: true,
  head: [
    ["meta", { property: "og:title", content: "TokenSpeed PKU Docs" }],
    [
      "meta",
      {
        property: "og:description",
        content:
          "Course results, reproducible evidence, and upstream runtime documentation."
      }
    ]
  ],
  themeConfig: {
    nav: [
      { text: "Docs Home", link: "/" },
      { text: "课程交付", link: "/project/final_delivery_20261018" },
      { text: "Getting Started", link: "/guides/getting-started" },
      { text: "Launching", link: "/guides/launching" },
      { text: "Recipes", link: "/recipes/models" },
      { text: "Configuration", link: "/configuration/server" },
      { text: "GitHub", link: "https://github.com/jiran-1111/tokenspeed_PKU" }
    ],
    sidebar: [
      {
        text: "Overview",
        items: [{ text: "Docs Home", link: "/" }]
      },
      {
        text: "课程项目",
        items: [
          { text: "最终交付与复现", link: "/project/final_delivery_20261018" },
          { text: "课程总报告", link: "/report/course_project_final_20261018" },
          { text: "公开证据", link: "/evidence/course_final_20261018/README" },
          { text: "答辩材料", link: "/presentation/README" },
          { text: "开源维护", link: "/project/oss_maintenance_20261018" }
        ]
      },
      {
        text: "Guides",
        items: [
          { text: "Getting Started", link: "/guides/getting-started" },
          { text: "Launching a Server", link: "/guides/launching" },
          { text: "InstantTensor Loading", link: "/guides/instanttensor" }
        ]
      },
      {
        text: "Configuration",
        items: [
          { text: "Server Parameters", link: "/configuration/server" },
          {
            text: "Compatible Parameters",
            link: "/configuration/compatible-parameters"
          }
        ]
      },
      {
        text: "Recipes",
        items: [{ text: "Model Recipes", link: "/recipes/models" }]
      },
      {
        text: "Serving",
        items: [{ text: "Parallelism", link: "/serving/parallelism" }]
      }
    ],
    search: {
      provider: "local"
    },
    editLink: {
      pattern:
        "https://github.com/jiran-1111/tokenspeed_PKU/edit/fix/oss-project-20261018/docs/:path",
      text: "Edit this page on GitHub"
    },
    footer: {
      message:
        "PKU course fork; runtime and design documentation retain upstream authorship.",
      copyright: "Copyright © lightseekorg"
    },
    socialLinks: [
      { icon: "github", link: "https://github.com/jiran-1111/tokenspeed_PKU" }
    ],
    outline: "deep",
    outlineTitle: "On this page",
    docFooter: {
      prev: "Previous page",
      next: "Next page"
    }
  }
});
