import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Enterprise AI Data Analyst",
  description: "Enterprise conversational analytics with Gemini & LangGraph",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body style={{ margin: 0, padding: 0, fontFamily: "sans-serif", backgroundColor: "#0b0f19", color: "#f3f4f6" }}>
        {children}
      </body>
    </html>
  );
}
