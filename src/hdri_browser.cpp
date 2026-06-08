// HdriBrowser.cpp — Async HDRI/EXR thumbnail browser panel for Dear ImGui.
//
// Workers load + downsample images on background threads.  The main thread
// drains the ready queue in uploadPending() (before beginFrame) and does the
// Vulkan upload there.  draw() is a pure ImGui call with no GPU work.

#include "hdri_browser.h"
#include "texture.h"

#include <imgui.h>
#include <nfd.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <string>

// ─── Construction / Destruction ───────────────────────────────────────────────

HdriBrowser::HdriBrowser()
{
    startWorkers();
}

HdriBrowser::~HdriBrowser()
{
    // GPU resources must have been freed by shutdown() already.
    stopWorkers();
}

// ─── Thread pool ─────────────────────────────────────────────────────────────

void HdriBrowser::startWorkers()
{
    m_stopWorkers = false;
    m_workers.reserve(THUMB_WORKERS);
    for (int i = 0; i < THUMB_WORKERS; ++i)
    {
        m_workers.emplace_back([this] { workerLoop(); });
    }
}

void HdriBrowser::stopWorkers()
{
    {
        std::lock_guard<std::mutex> lock(m_workMutex);
        m_stopWorkers = true;
        while (!m_workQueue.empty())
        {
            m_workQueue.pop();  // discard pending
        }
    }
    m_workCv.notify_all();
    for (auto& t : m_workers)
    {
        if (t.joinable())
        {
            t.join();
        }
    }
    m_workers.clear();
    m_stopWorkers = false;
}

void HdriBrowser::workerLoop()
{
    while (true)
    {
        WorkItem item;
        {
            std::unique_lock<std::mutex> lock(m_workMutex);
            m_workCv.wait(lock, [this]
            {
                return m_stopWorkers || !m_workQueue.empty();
            });
            if (m_stopWorkers && m_workQueue.empty())
            {
                return;
            }
            item = std::move(m_workQueue.front());
            m_workQueue.pop();
        }

        // Load the full image on this worker thread.
        // item.path is UTF-8; use u8path() so the dot-search in extension()
        // isn't confused by non-ASCII bytes on Windows.
        Texture tex;
        std::string err;
        std::string ext = std::filesystem::u8path(item.path).extension().string();
        for (char& c : ext)
        {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        const bool isHdr = (ext == ".hdr");
        const bool ok    = isHdr ? tex.loadHDR(item.path, err)
                                 : tex.loadEXR(item.path, err);

        ReadyItem ready;
        ready.idx = item.idx;

        if (!ok || !tex.isHdr() || tex.width == 0 || tex.height == 0)
        {
            ready.errorMsg = ok ? "Expected HDR (float) image" : err;
        }
        else
        {
            ready.pixels.resize(static_cast<size_t>(THUMB_W) * THUMB_H * 4u);
            generateThumbnail(tex.floatPixels(), tex.width, tex.height,
                              ready.pixels.data(), THUMB_W, THUMB_H);
        }

        {
            std::lock_guard<std::mutex> lock(m_readyMutex);
            m_readyQueue.push(std::move(ready));
        }
    }
}

// ─── Thumbnail generation ─────────────────────────────────────────────────────
// Runs on a worker thread.  Box-filter downsample + log-average auto-exposure
// + per-channel Reinhard tone-map + sRGB gamma encode → RGBA8.

void HdriBrowser::generateThumbnail(const float* src, int srcW, int srcH,
                                    uint8_t* dst, int dstW, int dstH)
{
    // ── Log-average luminance for auto-exposure ───────────────────────────────
    double logSum = 0.0;
    const int totalPx = srcW * srcH;
    for (int i = 0; i < totalPx; ++i)
    {
        const float lum = 0.2126f * src[i * 4 + 0]
                        + 0.7152f * src[i * 4 + 1]
                        + 0.0722f * src[i * 4 + 2];
        logSum += std::log(static_cast<double>(std::fmax(lum, 1e-4f)));
    }
    const float logAvgLum = static_cast<float>(std::exp(logSum / totalPx));
    const float exposure   = 0.18f / logAvgLum;  // key = 0.18 (middle grey)

    // ── Box-filter downsample ────────────────────────────────────────────────
    const float scaleX = static_cast<float>(srcW) / dstW;
    const float scaleY = static_cast<float>(srcH) / dstH;

    for (int dy = 0; dy < dstH; ++dy)
    {
        const int sy0 = static_cast<int>(dy       * scaleY);
        const int sy1 = std::min(static_cast<int>((dy + 1) * scaleY), srcH);

        for (int dx = 0; dx < dstW; ++dx)
        {
            const int sx0 = static_cast<int>(dx       * scaleX);
            const int sx1 = std::min(static_cast<int>((dx + 1) * scaleX), srcW);

            float r = 0.0f, g = 0.0f, b = 0.0f;
            int   n = 0;
            for (int sy = sy0; sy < sy1; ++sy)
            {
                for (int sx = sx0; sx < sx1; ++sx)
                {
                    const float* p = src + (static_cast<size_t>(sy) * srcW + sx) * 4;
                    r += p[0]; g += p[1]; b += p[2]; ++n;
                }
            }
            if (n > 0)
            {
                r /= n; g /= n; b /= n;
            }

            // Per-channel Reinhard with auto-exposure
            r = (r * exposure) / (1.0f + r * exposure);
            g = (g * exposure) / (1.0f + g * exposure);
            b = (b * exposure) / (1.0f + b * exposure);

            // sRGB gamma encode
            r = std::pow(std::fmax(r, 0.0f), 1.0f / 2.2f);
            g = std::pow(std::fmax(g, 0.0f), 1.0f / 2.2f);
            b = std::pow(std::fmax(b, 0.0f), 1.0f / 2.2f);

            uint8_t* out = dst + (static_cast<size_t>(dy) * dstW + dx) * 4;
            out[0] = static_cast<uint8_t>(std::clamp(r * 255.0f, 0.0f, 255.0f));
            out[1] = static_cast<uint8_t>(std::clamp(g * 255.0f, 0.0f, 255.0f));
            out[2] = static_cast<uint8_t>(std::clamp(b * 255.0f, 0.0f, 255.0f));
            out[3] = 255u;
        }
    }
}

// ─── Folder management ────────────────────────────────────────────────────────

void HdriBrowser::clearEntries(VulkanContext& vkCtx)
{
    stopWorkers();

    // Drain the ready queue before touching entries
    {
        std::lock_guard<std::mutex> lock(m_readyMutex);
        while (!m_readyQueue.empty())
        {
            m_readyQueue.pop();
        }
    }

    for (auto& e : m_entries)
    {
        vkCtx.destroyImGuiTexture(e.texture);
    }
    m_entries.clear();

    startWorkers();
}

void HdriBrowser::setFolder(VulkanContext& vkCtx, const std::string& folderPath)
{
    clearEntries(vkCtx);
    m_folder = folderPath;

    // Interpret folderPath as UTF-8 (NFD always returns UTF-8 on all platforms).
    // On Windows, std::filesystem::path(std::string) uses the ANSI code page, so
    // non-ASCII folder names would produce a broken path.  u8path() decodes the
    // bytes as UTF-8 and stores them as UTF-16 internally.
    const std::filesystem::path fPath = std::filesystem::u8path(folderPath);

    try
    {
        // skip_permission_denied: silently skip sub-folders we can't read instead
        // of aborting the whole scan with an exception.
        const auto iterOpts = std::filesystem::directory_options::skip_permission_denied;
        for (const auto& dirEntry :
             std::filesystem::recursive_directory_iterator(fPath, iterOpts))
        {
            // Use the no-ec overload so it reads from the cached directory-entry
            // status (fast, no extra syscall).  The ec overload on MSVC writes back
            // to the error_code after each call; if the same ec was passed to the
            // iterator constructor, a non-zero write here stops iteration early.
            if (!dirEntry.is_regular_file())
            {
                continue;
            }

            // Case-insensitive extension check — tolower handles .EXR / .Exr / etc.
            std::string ext = dirEntry.path().extension().string();
            for (char& c : ext)
            {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            if (ext != ".exr" && ext != ".hdr")
            {
                continue;
            }

            const int idx = static_cast<int>(m_entries.size());
            ThumbEntry te;
            // u8string() gives UTF-8 on every platform.  On Windows, .string()
            // would use the ANSI code page and corrupt paths containing ä/ö/å etc.
            te.path = dirEntry.path().u8string();
            // Show the path relative to the chosen root so files in different
            // sub-folders with the same filename are distinguishable in the grid.
            te.name = std::filesystem::relative(dirEntry.path(), fPath).u8string();
            te.state = ThumbState::Loading;
            m_entries.push_back(std::move(te));

            {
                std::lock_guard<std::mutex> lock(m_workMutex);
                m_workQueue.push({ idx, m_entries.back().path });
            }
        }
    }
    catch (const std::exception& /*e*/)
    {
        // Root directory not accessible — m_entries stays empty and the UI will
        // show "No .exr or .hdr files found in this folder."
    }

    m_workCv.notify_all();
}

// ─── Main-thread GPU upload ───────────────────────────────────────────────────

void HdriBrowser::uploadPending(VulkanContext& vkCtx)
{
    // Process a folder change queued by draw() on the previous frame.
    if (!m_pendingFolder.empty())
    {
        setFolder(vkCtx, m_pendingFolder);
        m_pendingFolder.clear();
    }

    // Drain the ready queue and upload all completed thumbnails to GPU.
    std::vector<ReadyItem> batch;
    {
        std::lock_guard<std::mutex> lock(m_readyMutex);
        while (!m_readyQueue.empty())
        {
            batch.push_back(std::move(m_readyQueue.front()));
            m_readyQueue.pop();
        }
    }

    for (auto& item : batch)
    {
        if (item.idx < 0 || item.idx >= static_cast<int>(m_entries.size()))
        {
            continue;
        }

        auto& entry = m_entries[item.idx];
        if (item.errorMsg.empty())
        {
            entry.texture = vkCtx.createImGuiTexture(
                THUMB_W, THUMB_H, item.pixels.data());
            entry.state = ThumbState::GPUReady;
        }
        else
        {
            entry.state    = ThumbState::Error;
            entry.errorMsg = item.errorMsg;
        }
    }
}

// ─── Shutdown ─────────────────────────────────────────────────────────────────

void HdriBrowser::shutdown(VulkanContext& vkCtx)
{
    stopWorkers();

    {
        std::lock_guard<std::mutex> lock(m_readyMutex);
        while (!m_readyQueue.empty())
        {
            m_readyQueue.pop();
        }
    }

    for (auto& e : m_entries)
    {
        vkCtx.destroyImGuiTexture(e.texture);
    }
    m_entries.clear();
}

// ─── ImGui panel ─────────────────────────────────────────────────────────────

bool HdriBrowser::draw(bool* open, std::string& selectedPath)
{
    ImGui::SetNextWindowSize(ImVec2(900.0f, 600.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("HDRI Browser", open))
    {
        ImGui::End();
        return false;
    }

    bool selected = false;

    // ── Toolbar ───────────────────────────────────────────────────────────────
    if (ImGui::Button("Select Folder..."))
    {
        nfdu8char_t* outPath = nullptr;
        if (NFD_PickFolderU8(&outPath, nullptr) == NFD_OKAY)
        {
            // Store the folder; setFolder() (which needs VulkanContext) is
            // called from uploadPending() at the start of the next frame.
            m_pendingFolder = reinterpret_cast<const char*>(outPath);
            NFD_FreePathU8(outPath);
        }
    }

    ImGui::SameLine();
    if (!m_folder.empty())
    {
        ImGui::TextUnformatted(m_folder.c_str());
    }
    else
    {
        ImGui::TextDisabled("No folder selected");
    }

    // Loading progress indicator
    {
        int nLoading = 0;
        for (const auto& e : m_entries)
        {
            if (e.state == ThumbState::Loading)
            {
                ++nLoading;
            }
        }
        if (nLoading > 0)
        {
            ImGui::SameLine();
            ImGui::TextDisabled("  (%d loading...)", nLoading);
        }
    }

    // ── Size selector — right-aligned on the same toolbar row ────────────────
    {
        static const char* const sizeLabels[] = { "Large", "Medium", "Small" };
        const float comboW = 90.0f;
        ImGui::SameLine(ImGui::GetContentRegionMax().x - comboW);
        ImGui::SetNextItemWidth(comboW);
        ImGui::Combo("##thumbsz", &m_thumbSizeIdx, sizeLabels, 3);
    }

    ImGui::Separator();

    // ── Thumbnail grid ────────────────────────────────────────────────────────
    if (m_entries.empty())
    {
        if (!m_folder.empty())
        {
            ImGui::TextDisabled("No .exr or .hdr files found in this folder.");
        }
        ImGui::End();
        return false;
    }

    // Display dimensions per size level (pixel data is always THUMB_W×THUMB_H).
    static const int kDispW[] = { THUMB_W, 192, 128 };
    static const int kDispH[] = { THUMB_H,  96,  64 };
    const int dispW = kDispW[m_thumbSizeIdx];
    const int dispH = kDispH[m_thumbSizeIdx];

    const float padding = 8.0f;
    const float labelH  = ImGui::GetTextLineHeightWithSpacing();

    // ImageButton adds FramePadding on each side of the image; account for it
    // so that cellW, cellH, and the label wrap position are all exact.
    const float fp  = ImGui::GetStyle().FramePadding.x;
    const float fpy = ImGui::GetStyle().FramePadding.y;
    const float cellW = static_cast<float>(dispW) + 2.0f * fp  + padding;
    const float cellH = static_cast<float>(dispH) + 2.0f * fpy + labelH + padding;

    ImGui::BeginChild("##thumbgrid", ImVec2(0.0f, 0.0f), false,
                      ImGuiWindowFlags_HorizontalScrollbar);

    // Query available width INSIDE the child so the column count accounts
    // for the child's own scrollbar reservation and inner padding.
    // Measuring in the parent window (before BeginChild) gives a wider value,
    // causing the rightmost column to overflow the child's clip rect.
    const float avail = ImGui::GetContentRegionAvail().x;

    // n items with (n-1) gaps: row = n*cellW - padding ≤ avail → n ≤ (avail+padding)/cellW.
    const int   cols  = std::max(1, static_cast<int>((avail + padding) / cellW));

    // thumbSz is the *image* size passed to ImageButton; the button's outer size
    // (and therefore the effective cell width) is thumbSz + 2 × FramePadding.
    const ImVec2 thumbSz (static_cast<float>(dispW),              static_cast<float>(dispH));
    // Error / Loading use Button with an explicit outer size that matches the
    // ImageButton outer size so all three states occupy the same cell width.
    const ImVec2 buttonSz(static_cast<float>(dispW) + 2.0f * fp,
                          static_cast<float>(dispH) + 2.0f * fpy);

    for (int i = 0; i < static_cast<int>(m_entries.size()); ++i)
    {
        auto& e = m_entries[i];

        if (i % cols != 0)
        {
            ImGui::SameLine(0.0f, padding);
        }

        ImGui::PushID(i);
        ImGui::BeginGroup();

        const bool isActive = !m_activePath.empty() && (e.path == m_activePath);

        if (e.state == ThumbState::GPUReady && e.texture.valid())
        {
            if (isActive)
            {
                ImGui::PushStyleColor(ImGuiCol_Button,
                                      ImVec4(0.20f, 0.50f, 0.90f, 0.60f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                      ImVec4(0.30f, 0.60f, 1.00f, 0.80f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                      ImVec4(0.10f, 0.40f, 0.80f, 1.00f));
            }

            if (ImGui::ImageButton("##img",
                    reinterpret_cast<ImTextureID>(e.texture.descSet), thumbSz))
            {
                selectedPath = e.path;
                m_activePath = e.path;
                selected     = true;
            }

            if (isActive)
            {
                ImGui::PopStyleColor(3);
            }

            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("%s", e.name.c_str());
            }
        }
        else if (e.state == ThumbState::Error)
        {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.40f, 0.10f, 0.10f, 1.0f));
            ImGui::Button("##err", buttonSz);
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("Error: %s\n%s", e.name.c_str(), e.errorMsg.c_str());
            }
        }
        else
        {
            // Loading placeholder — pulsing grey box
            const float t      = std::fmod(static_cast<float>(ImGui::GetTime()) * 1.5f,
                                           1.0f);
            const float bright = 0.12f + 0.08f * t;
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(bright, bright, bright, 1.0f));
            ImGui::Button("##load", buttonSz);
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("Loading: %s", e.name.c_str());
            }
        }

        // ── File name label ───────────────────────────────────────────────────
        // Wrap at the button's outer width (image + 2 × FramePadding).
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + buttonSz.x);
        if (isActive)
        {
            ImGui::TextColored(ImVec4(0.40f, 0.75f, 1.00f, 1.0f), "%s",
                               e.name.c_str());
        }
        else
        {
            ImGui::TextUnformatted(e.name.c_str());
        }
        ImGui::PopTextWrapPos();

        // Zero-size Dummy after the label.  Its purpose is purely to extend
        // the group's bounding-box max-y to include the trailing ItemSpacing.y
        // that follows the label.  Without it, EndGroup's ItemSize reports a
        // height that is one ItemSpacing.y short, making the row advance by
        // cellH - ItemSpacing.y instead of cellH.  A Dummy placed *outside*
        // EndGroup would overwrite CursorPosPrevLine.y and break SameLine's
        // row-start y for subsequent columns.
        ImGui::Dummy(ImVec2(0.0f, 0.0f));

        ImGui::EndGroup();
        ImGui::PopID();
    }

    ImGui::EndChild();
    ImGui::End();
    return selected;
}
