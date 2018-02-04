#Generate annotation for keras
#https://www.openu.ac.il/home/hassner/Adience/data.html#agegender

use warnings;
use strict;
use Image::Size;
use File::Copy;

my $dataset_path="dataset/";
#my $dataset_path="../yolo-keras-face-detection/dataset/agegender/";

my $ANNOTATION_FILES="./$dataset_path"."fold_";
my $FACE_FILES="./$dataset_path"."aligned";

mkdir "./local/faces/";

my $file_no=0;
my $line_no=0;

my $imagew;
my $imageh;

my $before_file="";

for(my $list=0;$list<5;$list=$list+1){
  open(IN,"$ANNOTATION_FILES"."$list"."_data.txt") or die ("agegender dataset not found");

  my $header=<IN>;

  while(my $line=<IN>){
    #print $line;

    my $user_id;
    my $original_image;
    my $face_id;
    my $age;
    my $gender;
    my $x;
    my $y;
    my $dx;
    my $dy;
    my $tilt_ang;
    my $fiducial_yaw_angle;
    my $fiducial_score;

    ($user_id,$original_image,$face_id,$age,$gender,$x,$y,$dx,$dy,$tilt_ang,$fiducial_yaw_angle,$fiducial_score)=split("\t",$line);

    $x=$x+$dx/2;
    $y=$y+$dy/2;
    my $w=$dx;
    my $h=$dy;
    
    my $category=-1;
    my $category_label="";
    my $age_int=-1;
    if($age =~ /^[0-9]+$/){
      $age_int=int($age);
    }else{
      if($age =~ /\(([0-9]+),/){
        $age_int=int("$1");
      }
    }

    my $thumb_dir="$FACE_FILES/$user_id/";

    opendir(THUMB, "$thumb_dir") or die "usage: $0 thumb_dir\n";
    my $filepath="";
    foreach my $dir (readdir(THUMB)) {
      next if ($dir eq '.' || $dir eq '..');
      next if ($dir eq '.DS_Store');
      if($dir =~ /$face_id\.$original_image/){
        $filepath=$dir;
        last;
      }
    }

    if($filepath eq ""){
      print "image file not found\n";
      next;
    }

    if($before_file ne $original_image){
      $before_file=$original_image;
      ($imagew, $imageh) = imgsize("$FACE_FILES/$user_id/$filepath");
      $file_no=$file_no+1;
      copy("$FACE_FILES/$user_id/$filepath","./local/faces/$filepath");
    }
    
    $x=1.0*$x/$imagew;
    $y=1.0*$y/$imagew;
    $w=1.0*$w/$imagew;
    $h=1.0*$h/$imagew;
  }

  close(IN);
}
