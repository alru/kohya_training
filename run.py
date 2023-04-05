import os
import argparse
import config
from install_packages_models import install_packages, install_extra_packages, download_custom_model, download_trained_model, download_vae


def run(args):
    #TODO check if empty value not to run
    assign_args_config(args=args)
    if config.install_package:
        install_extra_packages.run()
        install_packages.run()
    # We import those modules here as they depend on the installed packages in the last 2 lines, thereby can't import them before installing packages
    from data import get_data
    from data.data_processing import blib_caption, custom_caption, data_cleaning, waifu_diff_tagger
    from train_test_model import test,train
    from train_test_model.configurations import model_config, train_config, dataset_config, LoRA_Optim_config
    download_trained_model.install_checkpoint()
    download_custom_model.install()
    download_vae.install_vae()
    get_data.run()
    data_cleaning.run()
    if  config.custom_caption:
        custom_caption.run()
    else:
        blib_caption.run()
        waifu_diff_tagger.run()
    model_config.run()
    dataset_config.run()
    LoRA_Optim_config.run()
    train_config.run()
    train.run()
    test.main()




def assign_args_config(args):
    """This function takes the new defined arguments by command line and update the config values to the new ones"""
    config.install_package = args.install_package
    config.custom_caption = args.custom_caption
    config.extra_packages =args.extra_packages
    config.root_dir = args.root_dir
    config.deps_dir = args.deps_dir
    config.repo_dir = args.repo_dir
    config.training_dir = args.training_dir
    config.pretrained_model = args.pretrained_model
    config.config_dir = args.config_dir

    config.repo_url = args.repo_url
    config.bitsandytes_main_py = args.bitsandytes_main_py
    config.accelerate_config = args.accelerate_config
    config.tools_dir = args.tools_dir
    config.finetune_dir = args.finetune_dir

    config.branch = args.branch
    config.install_xformers = args.install_xformers
    config.mount_drive = args.mount_drive
    config.verbose = args.verbose

    config.modelName = args.modelName
    config.v2ModelName = args.v2ModelName

    config.modelUrl = args.modelUrl
    config.vaeName = args.vaeName

    config.train_data_dir = args.train_data_dir
    config.reg_data_dir = args.reg_data_dir

    config.zipfile_url = args.zipfile_url
    config.zipfile_name = args.zipfile_name
    config.unzip_to = args.unzip_to

    config.convert = args.convert
    config.random_color = args.random_color

    config.batch_size_bilb = args.batch_size_bilb
    config.max_data_loader_n_workers_bilb = args.max_data_loader_n_workers_bilb
    config.beam_search = args.beam_search
    config.min_length = args.min_length
    config.max_length = args.max_length

    config.batch_size_waifu = args.batch_size_waifu
    config.max_data_loader_n_workers_waifu = args.max_data_loader_n_workers_waifu

    config.model = args.model
    config.remove_underscores = args.remove_underscores
    config.threshold = args.threshold

    config.extension = args.extension
    config.custom_tag = args.custom_tag
    config.append = args.append
    config.remove_tag = args.remove_tag

    config.v2 = args.v2
    config.v_parameterization = args.v_parameterization
    config.project_name = args.project_name
    config.pretrained_model_name_or_path = args.pretrained_model_name_or_path
    config.vae = args.vae

    config.output_dir = args.output_dir
    config.output_to_drive = args.output_to_drive

    config.train_repeats = args.train_repeats
    config.reg_repeats = args.reg_repeats

    config.instance_token = args.instance_token
    config.class_token = args.class_token

    config.add_token_to_caption = args.add_token_to_caption

    config.resolution = args.resolution
    config.flip_aug = args.flip_aug

    config.caption_extension = args.caption_extension
    config.caption_dropout_rate = args.caption_dropout_rate

    config.caption_dropout_every_n_epochs = args.caption_dropout_every_n_epochs
    config.keep_tokens = args.keep_tokens

    config.network_category = args.network_category

    config.conv_dim = args.conv_dim
    config.conv_alpha = args.conv_alpha
    config.network_dim = args.network_dim
    config.network_alpha = args.network_alpha
    config.network_weight = args.network_weight


    config.optimizer_type = args.optimizer_type
    config.optimizer_args = args.optimizer_args

    config.train_unet = args.train_unet
    config.unet_lr = args.unet_lr

    config.train_text_encoder = args.train_text_encoder
    config.text_encoder_lr = args.text_encoder_lr
    config.unet_lr = args.unet_lr
    config.lr_scheduler = args.lr_scheduler
    config.lr_warmup_steps = args.lr_warmup_steps
    config.lr_scheduler_power = args.lr_scheduler_power

    config.lowram = args.lowram
    config.enable_sample_prompt = args.enable_sample_prompt

    config.sampler = args.sampler

    config.noise_offset = args.noise_offset
    config.num_epochs = args.num_epochs
    config.train_batch_size = args.train_batch_size
    config.mixed_precision = args.mixed_precision
    config.save_precision = args.save_precision
    config.save_n_epochs_type = args.save_n_epochs_type
    config.save_n_epochs_type_value = args.save_n_epochs_type_value

    config.save_model_as = args.save_model_as
    config.max_token_length = args.max_token_length

    config.clip_skip = args.clip_skip
    config.gradient_checkpointing = args.gradient_checkpointing

    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.seed = args.seed
    config.logging_dir = args.logging_dir
    config.prior_loss_weight = args.prior_loss_weight

    config.sample_prompt = args.sample_prompt
    config.config_file = args.config_file
    config.dataset_config = args.dataset_config
    config.network_weight = args.network_weight
    config.no_verbose = args.no_verbose
    config.hf_token = args.hf_token

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--install_package",type=bool, default=config.install_package)
    parser.add_argument("--custom_caption",type=bool, default=config.custom_caption)

    parser.add_argument("--extra_packages",type=str, default=config.extra_packages,help="Extra packages to install, takes a string, each package separated with a space")
    parser.add_argument("--root_dir",type=str, default=config.root_dir,help="Root directory where the code runs")
    parser.add_argument("--deps_dir",type=str, default=config.deps_dir)
    parser.add_argument("--repo_dir",type=str, default=config.repo_dir)
    parser.add_argument("--training_dir",type=str, default=config.training_dir)
    parser.add_argument("--pretrained_model",type=str, default=config.pretrained_model)
    parser.add_argument("--config_dir",type=str, default=config.config_dir)

    parser.add_argument("--repo_url",type=str, default=config.repo_url)
    parser.add_argument("--bitsandytes_main_py",type=str, default=config.bitsandytes_main_py)
    parser.add_argument("--accelerate_config",type=str, default=config.accelerate_config)
    parser.add_argument("--tools_dir",type=str, default=config.tools_dir)
    parser.add_argument("--finetune_dir",type=str, default=config.finetune_dir)

    parser.add_argument("--branch",type=str, default=config.branch)
    parser.add_argument("--install_xformers",type=bool, default=config.install_xformers)
    parser.add_argument("--mount_drive",type=bool, default=config.mount_drive)
    parser.add_argument("--verbose",type=bool, default=config.verbose)

    parser.add_argument("--modelName",type=str, default=config.modelName)
    parser.add_argument("--v2ModelName",type=str, default=config.v2ModelName)

    parser.add_argument("--modelUrl",type=str, default=config.modelUrl)
    parser.add_argument("--vaeName",type=str, default=config.vaeName)

    parser.add_argument("--train_data_dir",type=str, default=config.train_data_dir)
    parser.add_argument("--reg_data_dir",type=str, default=config.reg_data_dir)

    parser.add_argument("--zipfile_url",type=str, default=config.zipfile_url)
    parser.add_argument("--zipfile_name",type=str, default=config.zipfile_name)
    parser.add_argument("--unzip_to",type=str, default=config.unzip_to)

    parser.add_argument("--convert",type=bool, default=config.convert)
    parser.add_argument("--random_color",type=bool, default=config.random_color)

    parser.add_argument("--batch_size_bilb",type=int, default=config.batch_size_bilb)
    parser.add_argument("--max_data_loader_n_workers_bilb",type=int, default=config.max_data_loader_n_workers_bilb)
    parser.add_argument("--beam_search",type=bool, default=config.beam_search)
    parser.add_argument("--min_length",type=int, default=config.min_length)
    parser.add_argument("--max_length",type=int, default=config.max_length)

    parser.add_argument("--batch_size_waifu",type=int, default=config.batch_size_waifu)
    parser.add_argument("--max_data_loader_n_workers_waifu",type=int, default=config.max_data_loader_n_workers_waifu)

    parser.add_argument("--model",type=str, default=config.model)
    parser.add_argument("--remove_underscores",type=bool, default=config.remove_underscores)
    parser.add_argument("--threshold",type=float, default=config.threshold)

    parser.add_argument("--extension",type=str, default=config.extension)
    parser.add_argument("--custom_tag",type=str, default=config.custom_tag)
    parser.add_argument("--hf_token",type=str, default=config.hf_token)
    parser.add_argument("--append",type=bool, default=config.append)
    parser.add_argument("--remove_tag",type=bool, default=config.remove_tag)

    parser.add_argument("--v2",type=bool, default=config.v2)
    parser.add_argument("--v_parameterization",type=bool, default=config.v_parameterization)
    parser.add_argument("--project_name",type=str, default=config.project_name)
    parser.add_argument("--pretrained_model_name_or_path",type=str, default=config.pretrained_model_name_or_path)
    parser.add_argument("--vae",type=str, default=config.vae)

    parser.add_argument("--output_dir",type=str, default=config.output_dir)
    parser.add_argument("--output_to_drive",type=bool, default=config.output_to_drive)

    parser.add_argument("--train_repeats",type=int, default=config.train_repeats)
    parser.add_argument("--reg_repeats",type=int, default=config.reg_repeats)

    parser.add_argument("--instance_token",type=str, default=config.instance_token)
    parser.add_argument("--class_token",type=str, default=config.class_token)

    parser.add_argument("--add_token_to_caption",type=bool, default=config.add_token_to_caption)

    parser.add_argument("--resolution",type=int, default=config.resolution)
    parser.add_argument("--flip_aug",type=bool, default=config.flip_aug)

    parser.add_argument("--caption_extension",type=str, default=config.caption_extension)
    parser.add_argument("--caption_dropout_rate",type=float, default=config.caption_dropout_rate)

    parser.add_argument("--caption_dropout_every_n_epochs",type=int, default=config.caption_dropout_every_n_epochs)
    parser.add_argument("--keep_tokens",type=int, default=config.keep_tokens)

    parser.add_argument("--network_category",type=str, default=config.network_category)

    parser.add_argument("--conv_dim",type=int, default=config.conv_dim)
    parser.add_argument("--conv_alpha",type=int, default=config.conv_alpha)
    parser.add_argument("--network_dim",type=int, default=config.network_dim)
    parser.add_argument("--network_alpha",type=int, default=config.network_alpha)
    parser.add_argument("--network_weight",type=str, default=config.network_weight)


    parser.add_argument("--optimizer_type",type=str, default=config.optimizer_type)
    parser.add_argument("--optimizer_args",type=str, default=config.optimizer_args)

    parser.add_argument("--train_unet",type=bool, default=config.train_unet)
    parser.add_argument("--unet_lr",type=float, default=config.unet_lr)

    parser.add_argument("--train_text_encoder",type=bool, default=config.train_text_encoder)
    parser.add_argument("--text_encoder_lr",type=float, default=config.text_encoder_lr)
    parser.add_argument("--lr_scheduler",type=str, default=config.lr_scheduler)
    parser.add_argument("--lr_warmup_steps",type=int, default=config.lr_warmup_steps)
    parser.add_argument("--lr_scheduler_power",type=int, default=config.lr_scheduler_power)

    parser.add_argument("--lowram",type=bool, default=config.lowram)
    parser.add_argument("--enable_sample_prompt",type=bool, default=config.enable_sample_prompt)

    parser.add_argument("--sampler",type=str, default=config.sampler)

    parser.add_argument("--noise_offset",type=float, default=config.noise_offset)
    parser.add_argument("--num_epochs",type=int, default=config.num_epochs)
    parser.add_argument("--train_batch_size",type=int, default=config.train_batch_size)
    parser.add_argument("--mixed_precision",type=str, default=config.mixed_precision)
    parser.add_argument("--save_precision",type=str, default=config.save_precision)
    parser.add_argument("--save_n_epochs_type",type=str, default=config.save_n_epochs_type)
    parser.add_argument("--save_n_epochs_type_value",type=int, default=config.save_n_epochs_type_value)

    parser.add_argument("--save_model_as",type=str, default=config.save_model_as)
    parser.add_argument("--max_token_length",type=int, default=config.max_token_length)

    parser.add_argument("--clip_skip",type=int, default=config.clip_skip)
    parser.add_argument("--gradient_checkpointing",type=bool, default=config.gradient_checkpointing)

    parser.add_argument("--gradient_accumulation_steps",type=int, default=config.gradient_accumulation_steps)
    parser.add_argument("--seed",type=int, default=config.seed)
    parser.add_argument("--logging_dir",type=str, default=config.logging_dir)
    parser.add_argument("--prior_loss_weight",type=float, default=config.prior_loss_weight)

    parser.add_argument("--sample_prompt",type=str, default=config.sample_prompt)
    parser.add_argument("--config_file",type=str, default=config.config_file)
    parser.add_argument("--dataset_config",type=str, default=config.dataset_config)
    parser.add_argument("--no_verbose",type=bool, default=config.no_verbose)

    args = parser.parse_args()
    
    run(args)