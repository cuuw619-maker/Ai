                    &cell.w_keep,
                    &cell.u_keep,
                    &cell.b_keep,
                    &cell.w_write,
                    &cell.u_write,
                    &cell.b_write,
                    &cell.w_candidate,
                    &cell.u_candidate,
                    &cell.b_candidate,
                    &cell.w_out,
                    &cell.b_out,
                ]
                .iter()
                .flat_map(|p| p.grad.iter())
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>();
                (layer, sum.sqrt() as f32)
            })
            .collect::<Vec<_>>();

        let update = self.optimizer.step_with_stats(self.model.parameters_mut());
        self.state.step += 1;

        let layers = self
            .model
            .cells
            .iter()
            .enumerate()
            .map(|(layer, cell)| {
                let weight_sum = [
                    &cell.w_keep,
                    &cell.u_keep,
                    &cell.b_keep,
                    &cell.w_write,
                    &cell.u_write,
                    &cell.b_write,
                    &cell.w_candidate,
                    &cell.u_candidate,
                    &cell.b_candidate,
                    &cell.w_out,
                    &cell.b_out,
                ]
                .iter()
                .flat_map(|p| p.data.iter())
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>();
                let gradient_norm = gradient_layers
                    .iter()
                    .find(|(index, _)| *index == layer)
                    .map(|(_, value)| *value)
                    .unwrap_or(0.0);
                let memory_sum = memory
                    .get(layer)
                    .into_iter()
                    .flat_map(|values| values.iter())
                    .map(|v| (*v as f64) * (*v as f64))
                    .sum::<f64>();
                let (activation_mean, activation_min, activation_max) = self
                    .model
                    .layer_activation_stats()
                    .get(layer)
                    .copied()
                    .unwrap_or((0.0, 0.0, 0.0));
                let routing = self
                    .model
                    .routing_stats()
                    .get(layer)
                    .copied()
                    .unwrap_or_default();
                LayerTrainingStats {
                    layer,
                    activation_mean,
                    activation_min,
                    activation_max,
                    weight_norm: weight_sum.sqrt() as f32,
                    gradient_norm,
                    memory_norm: memory_sum.sqrt() as f32,
                    active_blocks: routing.active_blocks,
                    skipped_blocks: routing.total_blocks.saturating_sub(routing.active_blocks),
                    active_channels: routing.active_channels,
                    skipped_channels: routing.skipped_channels,
                    routing_entropy: routing.entropy,
                }
            })
            .collect::<Vec<_>>();
        let snapshot = ModelTrainingSnapshot {
            parameter_count: self.model.parameter_count(),
            checksum: self.model.weights_checksum(),
            gradient_magnitude: norm,
            updated_parameters: update.updated_parameters,
            average_update: update.average_absolute_update,
            max_update: update.max_absolute_update,
            layers,
        };
        let _ = events.send(TrainingEvent::ModelSnapshot(snapshot));

        let elapsed = now_ms().saturating_sub(started).max(1) as f64 / 1000.0;