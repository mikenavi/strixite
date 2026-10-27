#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>
namespace strix {
struct BoundarySnapshotPoint { int64_t pos; bool image; };
// Image anchors kept by default (capped by the total): a client keeping its newest 3 pages visible plus a turn's
// up to 6 new ones (two parallel doc views of 3 pages) hides images up to the 9th newest in one request.
inline constexpr int kDefaultBoundaryImageSnapshots = 9;
// Reserve a bounded bank for image anchors so a long text-only continuation
// cannot evict them. Other slots retain recent message starts. Boundaries are
// control-token positions, never a split inside a vision block. The predecessor
// of the image's message is deliberate: Pi moves a tool image into a user
// message, but when hiding it changes the preceding tool result as well. A
// client that keeps the image inside the tool message resumes one message
// earlier (its assistant tool call), which is still a valid, cheap point.
inline std::vector<BoundarySnapshotPoint> boundary_snapshot_plan(
    const std::vector<int32_t> &prompt, int32_t im_start, const std::vector<int64_t> &images,
    int64_t system_end, int64_t generation_start, int count, int image_count = -1) {
    if (count <= 0) return {};
    std::vector<int64_t> messages, anchors;
    for (int64_t p = 0; p < generation_start; ++p)
        if (prompt[(size_t)p] == im_start) messages.push_back(p);
    for (const int64_t image : images) {
        auto at = std::lower_bound(messages.begin(), messages.end(), image);
        if (at - messages.begin() < 2) continue;
        const int64_t p = *(at - 2);
        if (p > system_end) anchors.push_back(p);
    }
    std::sort(anchors.begin(), anchors.end());
    anchors.erase(std::unique(anchors.begin(), anchors.end()), anchors.end());
    const size_t keep_images = (size_t)std::min(count, image_count < 0 ? kDefaultBoundaryImageSnapshots : image_count);
    if (anchors.size() > keep_images) anchors.erase(anchors.begin(), anchors.end() - keep_images);
    std::vector<BoundarySnapshotPoint> out;
    for (const auto p : anchors) out.push_back({p, true});
    std::vector<int64_t> recent;
    for (const auto p : messages)
        if (p > system_end && !std::binary_search(anchors.begin(), anchors.end(), p)) recent.push_back(p);
    const size_t keep_recent = (size_t)count - keep_images;
    if (recent.size() > keep_recent) recent.erase(recent.begin(), recent.end() - keep_recent);
    for (const auto p : recent) out.push_back({p, false});
    std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) { return a.pos < b.pos; });
    return out;
}
} // namespace strix
