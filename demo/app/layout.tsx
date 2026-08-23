import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const sans = Geist({ variable: "--font-sans", subsets: ["latin"] });
const mono = Geist_Mono({ variable: "--font-mono", subsets: ["latin"] });
const title = "TickYantra — Tail Latency Under Control";
const description = "Interactive control-plane lab for SLO-aware SGLang inference.";
const socialImage = "https://raw.githubusercontent.com/RitwijParmar/TickYantra/main/demo/public/og.png";
export const metadata: Metadata = {
  title,
  description,
  icons: { icon: "/favicon.svg" },
  openGraph: { title, description, images: [{ url: socialImage, width: 1200, height: 630 }] },
  twitter: { card: "summary_large_image", title, description, images: [socialImage] },
};
export default function RootLayout({children}:{children:React.ReactNode}) { return <html lang="en"><body className={`${sans.variable} ${mono.variable}`}>{children}</body></html>; }
