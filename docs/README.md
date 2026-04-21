# Docs / Figures

Drop exported figures here when you want them to render inline in the root README:

- `architecture.png` — Figure 3.1 from the report (overall system architecture)
- `results_summary.png` — Table 4.7 exported as an image
- `demo_screenshot.png` — screenshot of the Gradio UI
- `demo.gif` — short screen recording of the enrol-new-person flow

Easy ways to produce them:

- Figures from the report — open the PDF, take screenshots of Figure 3.1 / Table 4.7 and save them here.
- Demo GIF — with the app running, use a tool like [LICEcap](https://www.cockos.com/licecap/) or [Kap](https://getkap.co/) to record ~10 seconds of the enrolment flow in Tab 3.

Once they're here, swap the heading of the root `README.md` for an image tag like:

```markdown
![Demo](docs/demo.gif)
```
