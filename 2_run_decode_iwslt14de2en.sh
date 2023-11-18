#!/usr/bin/env bash

./2_decode_iwslt14de2en.sh None &> errlog001-2_decode-baseline
./2_decode_iwslt14de2en.sh temperature &> errlog001-2_decode-temperature
./2_decode_iwslt14de2en.sh att_temp &> errlog001-2_decode-att_temp
./2_decode_iwslt14de2en.sh mh_att_temp &> errlog001-2_decode-mh_att_temp
./2_decode_iwslt14de2en.sh band_width_scaling &> errlog001-2_decode-band_width_scaling
./2_decode_iwslt14de2en.sh mh_band_width_scaling &> errlog001-2_decode-mh_band_width_scaling
./2_decode_iwslt14de2en.sh ad_att_temp &> errlog001-2_decode-ad_att_temp
./2_decode_iwslt14de2en.sh mh_ad_att_temp &> errlog001-2_decode-mh_ad_att_temp