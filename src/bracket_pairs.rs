pub mod tables;

pub use std::vec::Vec; //not int
pub use tables::{BidiClass, bidi_class, UNICODE_VERSION};
use BidiClass::*;
use std::char;	// not in

mod bracket_pair{
	use std::cmp::Ordering;

	#[derive(Debug, Copy, Clone)]
	pub struct BracketPair {
	    pub ich_opener: u8,
	    pub ich_closer: u8
	}

	impl Ord for BracketPair {
	    fn cmp(&self, other: &Self) -> Ordering {
	        self.ich_opener.cmp(&other.ich_opener)
	    }
	}

	impl PartialOrd for BracketPair {
	    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
	        Some(self.cmp(other))
	    }
	}

	impl PartialEq for BracketPair {
	    fn eq(&self, other: &Self) -> bool {
	        (self.ich_opener, self.ich_closer) == (other.ich_opener, other.ich_closer)
	    }
	}

	impl Eq for BracketPair { }

}

const NON: u8 = 0;
const OPEN: u8 = 1;
const CLOSE: u8 = 2;

fn locate_brackets(indexes: &[u8], pair_types: &[u8],
		 pair_values: &[u8]) -> Vec<bracket_pair::BracketPair>{
	let mut bracket_pair_list = Vec::<bracket_pair::BracketPair>::new();
	let mut stack_index = Vec::<u8>::new();
	let mut stack_char = Vec::<u8>::new();
	//let mut next_position_on_stack:i32 = -1;
	//next_position_on_stack = next_position_on_stack - 1;

	//println!("{},{}, {}", next_position_on_stack, stack_char.len(), stack_index.len());
	'outer_loop: for index in 0..indexes.len() {
		let case = pair_values[indexes[index] as usize];
		//print!("\nFor index:{} indexes[index]:, {}: pair_values[indexes[index]:{}",index, indexes[index], case);
		if case == OPEN {
			//println!("{}", next_position_on_stack);
			//next_position_on_stack = next_position_on_stack + 1;
        	stack_index.push(index as u8);
        	stack_char.push(indexes[index as usize] as u8);
        	//print!("-->Opening Bracket, {}", indexes[index]);
		  	}
	  	else if case == CLOSE {
			  	//print!("-->Closing Bracket, {}", indexes[index]);
			  	if stack_index.len()==0 { // no opening bracket exists
			  		continue 'outer_loop;
			  	}//search the stack
			  	for rev_index in (0..stack_index.len()).rev(){ //iterate down the stack
			  		if(pair_types[indexes[index as usize] as usize]==pair_types[stack_char[rev_index as usize] as usize]){ // match found
			  			let new_br_pair: bracket_pair::BracketPair = bracket_pair::BracketPair {ich_opener:stack_index[rev_index as usize], ich_closer:indexes[index as usize]}; //make a new BracketPair
			  			stack_index.remove(rev_index);
			  			stack_char.remove(rev_index);
			  			bracket_pair_list.push(new_br_pair);
			  			println!("!!Added to list[{}, {}]", new_br_pair.ich_opener, new_br_pair.ich_closer);
			  		}
		  		}
			}
		else if case == NON {
  			//print!("-->Not a Bracket, {}", indexes[index]);
	 	}
	}
	bracket_pair_list.sort();
	bracket_pair_list//return
}

pub fn is_strong_type_by_N0(class: tables::BidiClass) -> bool {
	class == R || class == L
}

pub fn return_strong_type_by_N0(index: u8, indexes: &[u8]) -> tables::BidiClass{
	let c=bidi_class(char::from_u32(indexes[index as usize] as u32).unwrap());
	match c {
		EN => R,
		AN => R,
		AL => R,
		R  => R,
		L  => L,
		_  => ON
	}
}

pub fn classify_pair_content(indexes: &[u8], curr_pair: 
	bracket_pair::BracketPair, dir_embed: tables::BidiClass) -> tables::BidiClass{
	let mut dir_opposite = ON;
	for pos in curr_pair.ich_opener+1..curr_pair.ich_closer{
		//println!("return_strong_type_by_N0({}) is ON? {}", indexes[pos as usize], return_strong_type_by_N0(pos, indexes)==ON)
		let dir_found = return_strong_type_by_N0(pos, indexes);
		if is_strong_type_by_N0(dir_found){
			if dir_found == dir_embed{
				return dir_embed;
			} 
			dir_opposite = dir_found;
		}
		
	}
	//Return Opposite direction, unless no opposite direction found
	dir_opposite
}

pub fn first_strong_class_before_pair(indexes: &[u8], curr_pair: bracket_pair::BracketPair) -> tables::BidiClass{
	let mut dir_before = ON;
	'for_loop: for index in (0..curr_pair.ich_closer).rev() {
		let dir_found = return_strong_type_by_N0(index, indexes);
		if dir_found != ON{
			dir_before = dir_found;
			break 'for_loop;
		}
	}
	dir_before
}

pub fn resolve_bracket_pair(indexes: &[u8], codes: &[u8], dir_embed: tables::BidiClass,
			 sos: &i8, curr_pair: bracket_pair::BracketPair){
	println!("Trying to resolve {}--{}", curr_pair.ich_opener, curr_pair.ich_closer);
	let mut set_direction = true;
	let mut dir_of_pair = classify_pair_content(indexes, curr_pair, dir_embed);
	if  dir_of_pair == ON {
		set_direction = false;
	}
	if dir_of_pair != dir_embed {
		let dir_before = first_strong_class_before_pair(indexes, curr_pair);
		if(dir_before == dir_embed || dir_before == ON){
			dir_of_pair = dir_embed
		}
	}
	if set_direction == true{
		set_dir_of_br_pair(codes, curr_pair, dir_of_pair);
	}
}

pub fn set_dir_of_br_pair( codes: &[u8], br_pair: bracket_pair::BracketPair, 
				dir_to_be_set: tables::BidiClass) -> &[u8] {
	//Insert into codes --> set codes[br_pair.ich_opener] = dir_to_be_set;
	//Insert into codes --> set codes[br_pair.ich_closer] = dir_to_be_set;
	let dir_str = match dir_to_be_set{
		R  => "R",
		L  => "L",
		ON => "ON",
		_  => "Error"
	};
	println!("Setting {}--{} as {}", br_pair.ich_opener, br_pair.ich_closer, dir_str);
	codes
}

pub fn resolve_all_paired_brackets(indexes: &[u8], codes: &[u8], pair_types: &[u8],
		pair_values: &[u8], sos: &i8, level: &i8) {
	let bracket_pair_list= locate_brackets(indexes, pair_values, pair_types);
	let dir_embed:tables::BidiClass = 
	if 1 == (level & 1) {
		R
	} else{
		L
	};
	for br_pair in bracket_pair_list{
		println!("resolving {}--{}",br_pair.ich_opener, br_pair.ich_closer);
		resolve_bracket_pair(indexes, codes, dir_embed, sos, br_pair);
	}
	
}

fn main(){
	//println!("x {}", x);
	//something(2);
	//					   [(, [, }, {, ], )];
	let indexes: [u8; 6] = [0, 1, 2, 3, 4, 5];
	//indexes[-1] = 1;
	// indexes[1] = 2;
	// something(&indexes);
	let codes: [u8; 11] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
	let pair_values:[u8; 6] = [1, 1, 2, 1, 2, 2];
	let pair_types:[u8; 6] =  [7, 9, 11, 11, 9, 7];
	let sos: i8 = 0;
	let level: i8 = 0;
	resolve_all_paired_brackets(&indexes, &codes, &pair_values, &pair_types, &sos, &level);
}