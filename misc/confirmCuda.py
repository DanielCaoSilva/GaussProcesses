import torch

x = torch.rand(5, 3)
print(x)

print(torch.cuda.is_available())

# Old Code Snippets to store
# found_kernel, path_history = search_for_min_BIC(
# 	kernel_list, kernel_operations, kernel_str_running,
# 	bic_values, data_compact, scaler_consts, 0, initial_learning_rate, 1000)

# print(found_kernel)
# print(path_history)

# exact_gp_obj = TrainTestPlotSaveExactGP(
# 	ExactGPModel,
# 	kernel=kernel_str_running,
# 	train_x=data_compact[0], train_y=data_compact[1], test_x=data_compact[2], test_y=data_compact[3],
# 	scaler_min=scaler_consts[1], scaler_max=scaler_consts[0],
# 	num_iter=5,
# 	lr=0.01,  # lr=0.0063, #lr=0.01,
# 	name=kernel_str_running,
# 	save_loss_values="save",
# 	use_scheduler=True)
# exact_gp_obj.run_train_test_plot_kernel(set_xlim=[0.96, 1])


# for i in range(2):
# 	new_kernel_best = kernel_str_running
# 	kernel_str_running, path_history = search_for_min_BIC(
# 		kernel_list, kernel_operations,
# 		new_kernel_best, bic_values,
# 		data_compact, scaler_consts,
# 		initial_lr=initial_learning_rate, epoch_iter=10)
# 	save_history.append(path_history)
# column_names = [
# 	'n', 'Kernel_Name', 'BIC', 'Hyper_Parameters', 'Kernel']
# bic_out_df = pd.DataFrame(save_history)#, columns=column_names)
# # bic_out_df.to_csv('bin_save_history.csv')
# # print(bic_out_df.iloc[:, 0:2])
# print(bic_out_df)

# Search through possible composite kernel combinations for a better BIC value
# while bic_values[-1] > -2500:
# 	for ops_index, iter_ops in enumerate(kernel_operations):
# 		for kernel_term_index, iter_kernel_terms in enumerate(kernel_list):
# 			if n == 0:
# 				kernel_str = kernel_str_running
# 			else:
# 				kernel_str = add_new_kernel_term(
# 					kernel_str_running, iter_kernel_terms, iter_ops)
# 			exact_gp = TrainTestPlotSaveExactGP(
# 				ExactGPModel,
# 				kernel=kernel_str,
# 				train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
# 				scaler_min=scaler_min, scaler_max=scaler_max,
# 				num_iter=1000,
# 				lr=initial_learning_rate,  # lr=0.0063, #lr=0.01,
# 				name=kernel_str,
# 				save_loss_values="save",
# 				use_scheduler=True)
# 			current_bic_value, hyper_values = exact_gp.run_train_test_plot_kernel(set_xlim=[0.96, 1])
# 			print("Iterations Number(n): ", n, "Learning Rate: ", initial_learning_rate)
# 			print("Kernel Structure (Old Best): ", kernel_str_running, "\n BIC: ", bic_values[-1])
# 			print("Kernel Structure (Current Trial): ", kernel_str, "\n BIC: ", current_bic_value)
# 			if current_bic_value < bic_values[-1]:
# 				bic_values.append(current_bic_value)
# 				kernel_str_running = kernel_str
# 			bic_save.append([
# 				n, kernel_str,
# 				current_bic_value,
# 				hyper_values,
# 				exact_gp.kernel])
# 			n += 1
# 			del exact_gp
# 			gc.enable()
# 			gc.collect()
# 			torch.cuda.empty_cache()