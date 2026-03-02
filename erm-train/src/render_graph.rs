//! Graph renderer for visualizing RouteGraph snapshots.
//!
//! Renders graph structure as SVG for animation frames.

use crate::graph_snapshot::GraphSnapshot;
use erm_core::error::{ErmError, ErmResult};

/// Render a graph snapshot to SVG string.
///
/// # Arguments
/// * `snapshot` — The graph snapshot to render
/// * `batch_idx` — Which batch item to visualize (0 to batch_size-1)
///
/// # Errors
/// Returns error if batch_idx is out of bounds.
pub fn render_snapshot_to_svg(snapshot: &GraphSnapshot, batch_idx: usize) -> ErmResult<String> {
    if batch_idx >= snapshot.graph.batch_size {
        return Err(ErmError::InvalidConfig(format!(
            "batch_idx {} out of bounds (batch_size={})",
            batch_idx, snapshot.graph.batch_size
        )));
    }

    let l = snapshot.graph.seq_len;
    let e = snapshot.graph.emax;

    // Canvas dimensions
    let width = 1200;
    let height = 400;
    let margin = 50;
    let graph_width = width - 2 * margin;
    let _graph_height = height - 2 * margin;

    // Position nodes horizontally
    let node_y = height / 2;
    let node_spacing = graph_width as f32 / (l as f32 - 1.0).max(1.0);

    // Calculate incoming pheromone per node for node sizing
    let mut node_incoming_phi = vec![0.0_f32; l];
    #[allow(clippy::needless_range_loop)]
    for li in 0..l {
        for ei in 0..e {
            let idx = snapshot.graph.idx(batch_idx, li, ei);
            if snapshot.graph.nbr_idx[idx] >= 0 {
                node_incoming_phi[li] += snapshot.graph.phi[idx];
            }
        }
    }

    let max_incoming = node_incoming_phi.iter().copied().fold(0.0_f32, f32::max);

    // Build SVG
    let mut svg = String::new();

    // Header with proper escaping
    svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    svg.push_str(&format!(
        "<svg width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
        width, height
    ));
    svg.push_str("<defs>\n");
    svg.push_str("  <marker id=\"arrowhead\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\">\n");
    svg.push_str("    <polygon points=\"0 0, 10 3.5, 0 7\" fill=\"#666\"/>\n");
    svg.push_str("  </marker>\n");
    svg.push_str("</defs>\n");

    // Background
    svg.push_str(&format!(
        "<rect width=\"{}\" height=\"{}\" fill=\"#1a1a2e\"/>\n",
        width, height
    ));

    // Title and stats
    let stats = snapshot.graph_stats();
    svg.push_str(&format!(
        "<text x=\"{}\" y=\"30\" font-family=\"sans-serif\" font-size=\"16\" fill=\"#eee\" text-anchor=\"middle\">\n",
        width / 2
    ));
    svg.push_str(&format!(
        "  Step: {} | Loss: {:.4} | Edges: {} | Avg φ: {:.3} | Deaths: {}\n",
        snapshot.step, snapshot.loss, stats.total_edges, stats.avg_phi, snapshot.deaths
    ));
    svg.push_str("</text>\n");

    // Draw edges first (so they're behind nodes)
    for li in 0..l {
        let dest_x = margin + (li as f32 * node_spacing) as i32;

        for ei in 0..e {
            let idx = snapshot.graph.idx(batch_idx, li, ei);
            let src = snapshot.graph.nbr_idx[idx];

            if src >= 0 && src < l as i32 {
                let src_idx = src as usize;
                let src_x = margin + (src_idx as f32 * node_spacing) as i32;
                let phi = snapshot.graph.phi[idx];
                let is_leader = snapshot.graph.leader_edge[idx];

                // Color based on pheromone (blue=low, red=high)
                let intensity = (phi / 2.0).min(1.0);
                let r = (intensity * 255.0) as u8;
                let g = ((1.0 - intensity) * 100.0) as u8;
                let b = ((1.0 - intensity) * 255.0) as u8;
                let color = format!("#{:02x}{:02x}{:02x}", r, g, b);

                // Stroke width based on pheromone
                let stroke_width = 1.0 + phi * 2.0;

                // Draw edge as curved line
                let control_y = if src_idx < li {
                    node_y - 50 - (li - src_idx) as i32 * 2
                } else {
                    node_y + 50 + (src_idx - li) as i32 * 2
                };

                let opacity = 0.3 + intensity * 0.7;

                if is_leader {
                    svg.push_str(&format!(
                        "<path d=\"M {} {} Q {} {}, {} {}\" stroke=\"{}\" stroke-width=\"{}\" fill=\"none\" marker-end=\"url(#arrowhead)\" stroke-dasharray=\"5,3\" opacity=\"{}\"/>\n",
                        src_x, node_y,
                        (src_x + dest_x) / 2, control_y,
                        dest_x, node_y,
                        color,
                        stroke_width,
                        opacity
                    ));
                } else {
                    svg.push_str(&format!(
                        "<path d=\"M {} {} Q {} {}, {} {}\" stroke=\"{}\" stroke-width=\"{}\" fill=\"none\" marker-end=\"url(#arrowhead)\" opacity=\"{}\"/>\n",
                        src_x, node_y,
                        (src_x + dest_x) / 2, control_y,
                        dest_x, node_y,
                        color,
                        stroke_width,
                        opacity
                    ));
                }
            }
        }
    }

    // Draw nodes
    #[allow(clippy::needless_range_loop)]
    for li in 0..l {
        let x = margin + (li as f32 * node_spacing) as i32;
        let incoming = node_incoming_phi[li];

        // Node radius based on incoming pheromone
        let radius = 3.0 + (incoming / max_incoming.max(1.0)) * 10.0;

        // Node color based on position (gradient from blue to purple)
        let hue = 220 + (li * 40 / l.max(1));
        let color = format!("hsl({}, 70%, 60%)", hue);

        svg.push_str(&format!(
            "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"{}\" stroke=\"#fff\" stroke-width=\"1\"/>\n",
            x, node_y, radius, color
        ));

        // Position label for every 10th node
        if li % 10 == 0 || li == l - 1 {
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"10\" fill=\"#aaa\" text-anchor=\"middle\">{}</text>\n",
                x, node_y + 25, li
            ));
        }
    }

    // Legend
    svg.push_str(&format!(
        "<g transform=\"translate({}, {})\">\n",
        margin,
        height - 60
    ));
    svg.push_str("<text x=\"0\" y=\"0\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#aaa\">— Follower edge</text>\n");
    svg.push_str(
        "<line x1=\"100\" y1=\"-4\" x2=\"140\" y2=\"-4\" stroke=\"#888\" stroke-width=\"2\"/>\n",
    );
    svg.push_str("<text x=\"0\" y=\"20\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#aaa\">- - Leader edge</text>\n");
    svg.push_str("<line x1=\"100\" y1=\"16\" x2=\"140\" y2=\"16\" stroke=\"#888\" stroke-width=\"2\" stroke-dasharray=\"5,3\"/>\n");
    svg.push_str("<text x=\"0\" y=\"40\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#888\">Color = pheromone φ (blue→red)</text>\n");
    svg.push_str("</g>\n");
    svg.push_str("</svg>\n");

    Ok(svg)
}

/// Save a snapshot as SVG file.
///
/// # Errors
/// Returns error if directory creation or file write fails.
pub fn save_snapshot_svg<P: AsRef<std::path::Path>>(
    snapshot: &GraphSnapshot,
    batch_idx: usize,
    output_dir: P,
) -> ErmResult<()> {
    let svg = render_snapshot_to_svg(snapshot, batch_idx)?;
    let dir = output_dir.as_ref();
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("frame_{:05}.svg", snapshot.step));
    std::fs::write(&path, svg)?;
    Ok(())
}

/// Render summary info as text.
#[must_use]
pub fn render_summary(snapshot: &GraphSnapshot) -> String {
    let stats = snapshot.graph_stats();
    format!(
        "Step {:5} | Loss: {:7.4} | Edges: {:4} | Avg φ: {:.3} | Max φ: {:.3} | Leader: {:3} | Follower: {:3} | Deaths: {:5}",
        snapshot.step,
        snapshot.loss,
        stats.total_edges,
        stats.avg_phi,
        stats.max_phi,
        stats.leader_edges,
        stats.follower_edges,
        snapshot.deaths
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_snapshot::GraphSnapshot;
    use erm_core::ants::AntState;
    use erm_core::config::ErmConfig;
    use erm_core::graph::RouteGraph;

    fn test_config() -> ErmConfig {
        ErmConfig {
            vocab_size: 100,
            seq_len: 16,
            batch_size: 2,
            hidden_dim: 64,
            num_blocks: 2,
            emax: 4,
            topk: 5,
            num_ants: 10,
            phi_init: 0.01,
            phi_max: 100.0,
            death_streak: 100,
            max_edits_per_step: 0.15,
            leader_fraction: 0.2,
            ..ErmConfig::default()
        }
    }

    #[test]
    fn test_render_snapshot_svg_basic() {
        let config = test_config();
        let mut graph = RouteGraph::new(&config);

        // Add some test edges
        graph.nbr_idx[0] = 0;
        graph.phi[0] = 0.5;
        graph.nbr_idx[1] = 2;
        graph.phi[1] = 1.0;

        let ant_state = AntState::new(&config);
        let snapshot = super::super::graph_snapshot::GraphSnapshot::new(
            100, 3.5, 20, 0.5, 2, 5, 100, graph, ant_state,
        );

        let svg = render_snapshot_to_svg(&snapshot, 0).expect("render should succeed");

        // Basic SVG structure checks
        assert!(svg.contains("<?xml version"));
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("Step: 100"));
        assert!(svg.contains("Loss: 3.5000"));
    }

    #[test]
    fn test_render_snapshot_batch_idx_bounds() {
        let config = test_config();
        let graph = RouteGraph::new(&config);
        let ant_state = AntState::new(&config);
        let snapshot = super::super::graph_snapshot::GraphSnapshot::new(
            0, 0.0, 0, 0.0, 0, 0, 0, graph, ant_state,
        );

        // Valid batch_idx
        assert!(render_snapshot_to_svg(&snapshot, 0).is_ok());
        assert!(render_snapshot_to_svg(&snapshot, 1).is_ok());

        // Invalid batch_idx
        assert!(render_snapshot_to_svg(&snapshot, 2).is_err());
    }

    #[test]
    fn test_render_summary() {
        let config = test_config();
        let graph = RouteGraph::new(&config);
        let ant_state = AntState::new(&config);
        let snapshot = super::super::graph_snapshot::GraphSnapshot::new(
            100, 3.5, 20, 0.5, 2, 5, 100, graph, ant_state,
        );

        let summary = render_summary(&snapshot);
        assert!(summary.contains("Step   100"));
        assert!(summary.contains("Loss:"));
    }
}
