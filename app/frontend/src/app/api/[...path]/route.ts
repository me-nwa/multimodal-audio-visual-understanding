import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 300;

const BACKEND =
  process.env.API_PROXY_TARGET ?? "http://127.0.0.1:8000";

const UNAVAILABLE = {
  detail:
    "Cannot reach the FastAPI backend at http://127.0.0.1:8000. From the repo root run: uvicorn app.backend.main:app --host 127.0.0.1 --port 8000",
};

async function proxy(
  request: NextRequest,
  path: string[]
) {
  const target = `${BACKEND}/${path.join("/")}${request.nextUrl.search}`;
  const headers = new Headers();
  const contentType = request.headers.get("content-type");

  if (contentType) {
    headers.set("content-type", contentType);
  }

  try {
    const response = await fetch(target, {
      method: request.method,
      headers,
      body:
        request.method === "GET" || request.method === "HEAD"
          ? undefined
          : request.body,
      duplex: "half",
    } as RequestInit);

    return new NextResponse(response.body, {
      status: response.status,
      headers: {
        "content-type":
          response.headers.get("content-type") ?? "application/json",
      },
    });
  } catch {
    return NextResponse.json(UNAVAILABLE, { status: 503 });
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  return proxy(request, path);
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  return proxy(request, path);
}
