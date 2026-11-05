// Inside saved_current's case loop AFTER unchanged baseline production parity.
// This is a second fresh actual production composition, not a CPU candidate.
      if(foreground_replay) {
        ASSERT_TRUE(entry.contains("foreground_edge_reference"));const auto &reference=entry.at("foreground_edge_reference");
        const auto &rb=reference.at("native_report");
        const auto ref_size=rb.contains("bytes")?rb.at("bytes").get<std::size_t>():rb.at("byte_count").get<std::size_t>();
        const auto ref_bytes=bound_bytes(rb,ref_size);const auto ref_report=json::parse(ref_bytes.begin(),ref_bytes.end());
        const auto ref_name=reference.at("native_case_name").get<std::string>();ASSERT_EQ(ref_name,entry.at("name").get<std::string>());
        const json *ref_case=nullptr;
        for(const auto &rc:ref_report.at("cases"))if(rc.at("name")==ref_name){ASSERT_EQ(ref_case,nullptr);ref_case=&rc;}
        ASSERT_NE(ref_case,nullptr);const auto &rd=ref_case->at("private_soft_layer_diagnostics");
        ASSERT_EQ(rd.at("source_closure").get<std::string>(),"da5b6b9756a434e069c648bac9feee9e52bf9fa8b2ee92d1f2bb4cc7e13a5125");
        ASSERT_TRUE(rd.at("bound_inputs_byteexact_after_probe").get<bool>());ASSERT_TRUE(rd.at("source_closure_unchanged_after_probe").get<bool>());
        ASSERT_FALSE(rd.at("kernel_only_input_copies").get<bool>());ASSERT_FALSE(rd.at("candidate_is_production_mask").get<bool>());
        ASSERT_EQ(reference.at("union_matte"),ref_case->at("artifacts").at("soft_layer_union_M.u32"));
        for(auto i=entry.at("artifacts").begin();i!=entry.at("artifacts").end();++i)
          ASSERT_EQ(ref_case->at("input_bindings").at(i.key()),i.value())<<i.key();
        EXPECT_EQ(bound_bytes(ref_case->at("artifacts").at("selected_M.u32"),pixels*4u),arrays.at("selected_M.u32"));
        const auto expected_fg=bound_bytes(reference.at("union_matte"),pixels*4u);
        const auto expected_fg_words=words(expected_fg);ASSERT_EQ(expected_fg_words.size(),pixels);
        for(std::size_t i=0u;i<pixels;++i){ASSERT_LE(expected_fg_words[i],1u);ASSERT_GE(expected_fg_words[i],actual[i]);}

        auto candidate_program=effective_composition;
        for(UINT i=0u;i<foreground_edge_shaders_.size();++i)candidate_program.outline.foreground_edge_shaders[i]=foreground_edge_shaders_[i].Get();
        composition::resources_t fg_owner;fg_owner.set_recorded_execution(true);
        ASSERT_TRUE(fg_owner.ensure(device_.Get(),shape[0],shape[1],shape[2],shape[3],error_))<<error_;
        ASSERT_TRUE(fg_owner.queue(context_.Get(),candidate_program,request,error_))<<error_;
        const auto fg=fg_owner.bundle();ASSERT_TRUE(fg.all_stages_issued);
        auto fi=in;
        fi.current_cb96=fg.current.actual_contour_constants96.Get();fi.actual_root_args=fg.current.actual_final256.Get();
        fi.legacy_matte=fg.current.original_legacy_matte.Get();fi.sct=fg.current.sct.Get();
        fi.baseline_matte=fg.baseline_matte.Get();fi.numeric_proof=fg.numeric_proof.Get();
        fi.source_envelope104=fg.source.uip_envelope.Get();fi.cbc_meta512=fg.enclosure.receipt.Get();
        fi.raw_meta512=fg.ring.receipt.Get();fi.f2_data=fg.outline.calibration_data.Get();fi.f2_matte=fg.outline.matte.Get();
        selector::resources_t fg_selected;ASSERT_TRUE(fg_selected.ensure(device_.Get(),shape[0],shape[3],error_))<<error_;
        ASSERT_TRUE(fg_selected.queue(context_.Get(),programs.selector_program(),fi,error_))<<error_;
        const auto fv=fg_selected.views();
        std::map<std::string,std::vector<char>> candidate_arrays{
          {"CBC.u32",read(fg.enclosure.receipt.Get())},{"Raw.u32",read(fg.ring.receipt.Get())},
          {"F2.u32",read(fg.outline.calibration_data.Get())},{"BODY.u32",read(fg.enclosure.body.Get())},
          {"CBC_M.u32",read(fg.enclosure.matte.Get())},{"Raw_M.u32",read(fg.ring.matte.Get())},
          {"F2_M.u32",read(fg.outline.matte.Get())},{"selected_M.u32",read(fv.matte.Get())},
          {"selected_receipt.u32",read(fv.receipt.Get())},{"source_proof.u32",read(fg.source.source_proof.Get())},
          {"source_envelope.u32",read(fg.source.uip_envelope.Get())},{"numeric_proof.u32",read(fg.numeric_proof.Get())}};
        bool upstream_equal=true;
        for(const auto *key:{"CBC.u32","Raw.u32","BODY.u32","CBC_M.u32","Raw_M.u32","selected_receipt.u32","source_proof.u32","source_envelope.u32","numeric_proof.u32"}) {
          const bool same=candidate_arrays.at(key)==arrays.at(key);upstream_equal=upstream_equal&&same;
          EXPECT_TRUE(same)<<"FG modified immutable producer/source operand "<<key;
        }
        const bool source_equal=read(fg.source.reference_rgb8.Get())==read(bundle.source.reference_rgb8.Get())&&
          read(fg.source.analysis_band.Get())==read(bundle.source.analysis_band.Get());
        EXPECT_TRUE(source_equal);upstream_equal=upstream_equal&&source_equal;
        EXPECT_EQ(candidate_arrays.at("F2_M.u32"),candidate_arrays.at("selected_M.u32"));
        EXPECT_EQ(candidate_arrays.at("selected_M.u32"),expected_fg)<<"Actual port must equal the bound da5b GPU union";
        const auto old_f2=words(arrays.at("F2.u32")),new_f2=words(candidate_arrays.at("F2.u32"));
        const auto new_m=words(candidate_arrays.at("selected_M.u32")),raw_m=words(arrays.at("Raw_M.u32")),code1=words(arrays.at("CBC_M.u32"));
        ASSERT_EQ(old_f2.size(),68608u);ASSERT_EQ(new_f2.size(),old_f2.size());ASSERT_EQ(new_m.size(),pixels);
        UINT n0=0u,n1=0u,nc=0u;std::size_t additions=0u,withdrawals=0u;
        for(std::size_t i=0u;i<pixels;++i){EXPECT_LE(new_m[i],1u);EXPECT_LE(actual[i],new_m[i]);if(actual[i]!=0u&&new_m[i]==0u)++withdrawals;if(actual[i]==0u&&new_m[i]==1u){++additions;if(raw_m[i]==0u)++n0;else ++n1;if(code1[i]!=0u)++nc;}}
        EXPECT_GE(old_f2[73u],n1);EXPECT_GE(old_f2[76u],nc);
        EXPECT_EQ(new_f2[7u],old_f2[7u]+n0);EXPECT_EQ(new_f2[70u],old_f2[70u]+n0);
        EXPECT_EQ(new_f2[73u],old_f2[73u]-n1);EXPECT_EQ(new_f2[76u],old_f2[76u]-nc);
        for(std::size_t i=0u;i<new_f2.size();++i)if(i!=7u&&i!=70u&&i!=73u&&i!=76u)EXPECT_EQ(new_f2[i],old_f2[i])<<"F2 word "<<i;

        // A fresh public F2 mirror on actual Source5 operands must match the
        // full FG composition before its owned snapshot is interpreted.
        auto mirror_program=candidate_program.outline;
        mirror_program.enclosure.current_sct=fg.current.sct.Get();mirror_program.enclosure.current_uip=fg.source.uip_envelope.Get();
        auto &mp=mirror_program.enclosure.producer;
        mp.original_source=fg.source.reference_rgb8.Get();mp.transfer=enclosure::transfer_t::encoded_sdr;mp.input_known=false;mp.direct_encoded_recipe=5u;
        mp.postprocess_mode=fg.current.postprocess_mode;mp.expected_work=fg.current.expected_work;
        mp.transaction_token=fg.current.token;mp.model_frame=fg.current.frame;mp.model_observation_us=fg.current.observation_us;
        mp.analysis_generation=fg.current.analysis_generation;mp.analysis_roi={0u,0u,shape[0],shape[1]};
        ring::resources_t mirror;ASSERT_TRUE(mirror.ensure(device_.Get(),shape[0],shape[3],error_))<<error_;
        mirror.set_recorded_execution(false);
        ASSERT_TRUE(mirror.dispatch(context_.Get(),candidate_program.colour,candidate_program.components,
          fg.current.actual_contour_constants96.Get(),fg.source.analysis_band.Get(),fg.current.probability.Get(),fg.current.support.Get(),fg.current.ocr_record.Get(),
          fg.current.actual_final256.Get(),240u,error_,&mirror_program))<<error_;
        const auto me=mirror.enclosure_views();const auto mr=mirror.raw_views();const auto mf=mirror.mutable_views();
        EXPECT_EQ(read(me.receipt.Get()),candidate_arrays.at("CBC.u32"));EXPECT_EQ(read(mr.receipt.Get()),candidate_arrays.at("Raw.u32"));
        EXPECT_EQ(read(mf.calibration_data.Get()),candidate_arrays.at("F2.u32"));EXPECT_EQ(read(mf.matte.Get()),candidate_arrays.at("F2_M.u32"));
        ASSERT_NE(mirror.foreground_edge_original_mask(),nullptr);
        const auto original_posthole=read(mirror.foreground_edge_original_mask());EXPECT_EQ(original_posthole,arrays.at("selected_M.u32"));
        arrays.emplace("foreground_edge_post_hole_original_M.u32",original_posthole);
        for(const auto &[key,bytes]:candidate_arrays)arrays.emplace("foreground_edge_"+key,bytes);
        arrays.emplace("foreground_edge_expected_da5b_M.u32",expected_fg);
        foreground_case={{"source_closure",foreground_edge_closure_},{"reference",reference},
          {"actual_full_composition_and_selector",true},{"actual_F2_mirror_parity_required",true},
          {"baseline_OFF_literal_production_parity_required",true},{"immutable_post_hole_snapshot_equals_baseline_M",original_posthole==arrays.at("selected_M.u32")},
          {"actual_selected_M_equals_da5b_GPU_reference",candidate_arrays.at("selected_M.u32")==expected_fg},
          {"unique_additions",additions},{"withdrawals",withdrawals},{"Raw0_additions",n0},{"Raw1_recoveries",n1},{"Code1_recoveries",nc},
          {"counter_words",{7u,70u,73u,76u}},{"source_RGB_and_upstream_models_readonly",upstream_equal},
          {"scope","Fresh actual canonical FG production composition+selector; baseline arrays stay OFF; FG arrays are explicitly prefixed. No CPU candidate or renderer."}};
      }
